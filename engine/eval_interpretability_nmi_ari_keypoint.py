import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from tqdm import tqdm
from utils.get_landmark_coordinates import landmark_coordinates


def create_centers(data_loader, model, num_parts, num_landmarks, device=torch.device("cuda")):
    """
    Generate the center coordinates as tensor for the current model.
    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Data loader for the current data split.
    model: torch.nn.Module
        Net that generates assignment maps.
    num_parts: int
        Number of predicted parts
    num_landmarks: int
        Number of landmarks in the dataset
    device: torch.device
        Device to run the evaluation on
    Returns
    ----------
    centers_tensor: torch.cuda.FloatTensor, [data_size, num_parts * 2]
        Geometric centers for assignment maps of the whole dataset.
        The coordinate order is (col_1, row_1, ..., col_K, row_K)
    annos_tensor: torch.cuda.FloatTensor, [batch_size, num_landmarks * 2]
        Landmark coordinate annotations of the whole dataset.
        The coordinate order is (col_1, row_1, ..., col_K, row_K)
    active_tensor: torch.cuda.FloatTensor, [data_size, num_parts * 2]
        Contains for each predicted part whether it's active or not. If a part
        has a maximum attention map value of > 0.5 somewhere in the image,
        we denote it as active.
    """
    # tensor for collecting centers, labels, existence masks
    centers_collection = []
    annos_collection = []
    masks_collection = []
    active_parts_collection = []

    gt_labels = []
    pred_final = []
    pred_landmarks = []
    present_landmarks = []

    # iterating the data loader, landmarks shape: [N, num_landmarks, 4], column first
    # bbox shape: [N, 5]
    for i, (input_raw, gt_class, landmarks_raw, bbox_raw) in enumerate(data_loader):
        # to device
        input_raw = input_raw.to(device, non_blocking=True)
        landmarks_raw = landmarks_raw.to(device, non_blocking=True)
        bbox_raw = bbox_raw.to(device, non_blocking=True)

        # cut the input and transform the landmark
        inputs, landmarks_full, bbox = input_raw, landmarks_raw, bbox_raw

        # gather the landmark annotations, center outputs and existence masks
        with torch.inference_mode():
            # generate assignment map
            _, assignment, logits_parts, _ = model(inputs)
            logits = logits_parts.mean(dim=-1)
            assignment = assignment[:, :-1, :, :]
            maxes = assignment.max(-1)[0].max(-1)[0]
            active_parts = torch.where(maxes > 0.5, 1, 0).unsqueeze(-1).expand(-1, -1, 2)

            # calculate the center coordinates of shape [N, num_parts, 2]
            loc_x, loc_y, grid_x, grid_y = landmark_coordinates(
                assignment.cpu())
            x_centroid = loc_x * inputs.shape[-2] / assignment.shape[-2]
            y_centroid = loc_y * inputs.shape[-1] / assignment.shape[-1]
            centers = torch.stack((y_centroid, x_centroid), dim=-1).to(device, non_blocking=True)

            # extract the landmark and existence mask, [N, num_landmarks, 2]
            landmarks = landmarks_full[:, :, -3:-1]
            masks = landmarks_full[:, :, -1].unsqueeze(2).expand_as(landmarks)

            # normalize the coordinates with the bounding boxes
            bbox = bbox.unsqueeze(2)
            centers[:, :, 0] = (centers[:, :, 0] - bbox[:, 1]) / bbox[:, 3]
            centers[:, :, 1] = (centers[:, :, 1] - bbox[:, 2]) / bbox[:, 4]
            landmarks[:, :, 0] = (landmarks[:, :, 0] - bbox[:, 1]) / bbox[:, 3]
            landmarks[:, :, 1] = (landmarks[:, :, 1] - bbox[:, 2]) / bbox[:, 4]

            # collect the centers, annotations and masks
            centers_collection.append(centers)
            annos_collection.append(landmarks)
            masks_collection.append(masks)
            active_parts_collection.append(active_parts)

            gt_labels.append(gt_class.unsqueeze(0))
            pred_final.append(logits.argmax().unsqueeze(0))
            pred_landmarks.append(logits_parts[:, :, :-1].argmax(1))
            present_landmarks.append(assignment.max(-1)[0].max(-1)[0].view(-1))

    # list into tensors
    centers_tensor = torch.cat(centers_collection, dim=0)
    annos_tensor = torch.cat(annos_collection, dim=0)
    masks_tensor = torch.cat(masks_collection, dim=0)
    active_tensor = torch.cat(active_parts_collection, dim=0)

    gt_labels = torch.cat(gt_labels, dim=0)
    pred_final = torch.cat(pred_final, dim=0)
    present_landmarks = torch.cat(present_landmarks, dim=0)

    # reshape the tensors
    centers_tensor = centers_tensor.contiguous().view(centers_tensor.shape[0], num_parts * 2)
    annos_tensor = annos_tensor.contiguous().view(centers_tensor.shape[0], num_landmarks * 2)
    masks_tensor = masks_tensor.contiguous().view(centers_tensor.shape[0], num_landmarks * 2)
    active_tensor = active_tensor.contiguous().view(active_tensor.shape[0], num_parts * 2)

    return centers_tensor, annos_tensor, masks_tensor, active_tensor, \
        gt_labels, pred_final, pred_landmarks, present_landmarks


def distance_l2(prediction, annotation):
    """
    Average L2 distance of two numpy arrays.
    Parameters
    ----------
    prediction: np.array, [data_size, 1, 2]
        Landmark prediction.
    annotation: np.array, [data_size, 1, 2]
        Landmark annotation.
    Returns
    ----------
    error: float
        Average L2 distance between prediction and annotation.
    """
    diff_sq = (prediction - annotation) * (prediction - annotation)
    L2_dists = np.sqrt(np.sum(diff_sq, axis=2))
    error = np.mean(L2_dists)
    return error


def eval_nmi_ari(net, data_loader, dataset="cub", device=torch.device("cuda")):
    if dataset == "cub":
        return eval_nmi_ari_cub(net, data_loader, device)
    elif dataset == "part_imagenet":
        return eval_nmi_ari_part_imagenet(net, data_loader, device)
    else:
        raise ValueError("Dataset not supported.")


def eval_nmi_ari_part_imagenet(net, data_loader, device):
    """
    Get Normalized Mutual Information, Adjusted Rand Index for given method
    Parameters
    ----------
    net: torch.nn.Module
        The trained net to evaluate
    data_loader: torch.utils.data.DataLoader
        The dataset to evaluate
    device: torch.device
        The device to run the evaluation on
    Returns
    ----------
    nmi: float
        Normalized Mutual Information between predicted parts and gt parts as %
    ari: float
        Adjusted Rand Index between predicted parts and gt parts as %
    """
    all_nmi_preds_w_bg = []
    all_nmi_gts = []

    # iterating the data loader, landmarks shape: [N, num_landmarks, 4], column first
    # bbox shape: [N, 5]
    for (input_raw, _, landmarks_raw) in tqdm(data_loader, desc="Evaluating NMI/ARI"):
        batch_size = input_raw.shape[0]
        # to device
        input_raw = input_raw.to(device, non_blocking=True)
        landmarks_raw = landmarks_raw.to(device, non_blocking=True)

        # cut the input and transform the landmark
        inputs, landmarks_full = input_raw, landmarks_raw

        # Used to filter out all pixels that have < 0.1 value for all GT parts
        background_landmark = torch.full(size=(batch_size, 1, landmarks_full.shape[-2], landmarks_full.shape[-1]),
                                         fill_value=0.1).to(device, non_blocking=True)
        landmarks_full = torch.cat((landmarks_full, background_landmark), dim=1)

        # Check which part is most active per pixel
        landmarks_vec = torch.argmax(landmarks_full, dim=1)

        with torch.inference_mode():
            # generate assignment map
            maps = net(inputs)[1]
            part_name_mat_w_bg = F.interpolate(maps, size=inputs.shape[-2:], mode='bilinear', align_corners=False)

            pred_parts_loc_w_bg = torch.argmax(part_name_mat_w_bg, dim=1)
            all_nmi_preds_w_bg.append(pred_parts_loc_w_bg.cpu().numpy())
            all_nmi_gts.append(landmarks_vec.cpu().numpy())

    nmi_preds = np.concatenate(all_nmi_preds_w_bg, axis=0).flatten()
    nmi_gts = np.concatenate(all_nmi_gts, axis=0).flatten()

    nmi = normalized_mutual_info_score(nmi_gts, nmi_preds) * 100
    ari = adjusted_rand_score(nmi_gts, nmi_preds) * 100
    return nmi, ari


def eval_nmi_ari_cub(net, data_loader, device):
    """
    Get Normalized Mutual Information, Adjusted Rand Index for given method
    Parameters
    ----------
    net: torch.nn.Module
        The trained net to evaluate
    data_loader: torch.utils.data.DataLoader
        The dataset to evaluate
    device: torch.device
        The device to run the evaluation on
    Returns
    ----------
    nmi: float
        Normalized Mutual Information between predicted parts and gt parts as %
    ari: float
        Adjusted Rand Index between predicted parts and gt parts as %
    """
    all_nmi_preds_w_bg = []
    all_nmi_gts = []

    # iterating the data loader, landmarks shape: [N, num_landmarks, 4], column first
    # bbox shape: [N, 5]
    for (input_raw, gt_class, landmarks_raw, bbox_raw) in tqdm(data_loader, desc="Evaluating NMI/ARI"):
        # to device
        input_raw = input_raw.to(device, non_blocking=True)
        landmarks_raw = landmarks_raw.to(device, non_blocking=True)
        bbox_raw = bbox_raw.to(device, non_blocking=True)

        # cut the input and transform the landmark
        inputs, landmarks_full, bbox = input_raw, landmarks_raw, bbox_raw

        with torch.inference_mode():
            # generate assignment map
            maps = net(inputs)[1]

            part_name_mat_w_bg = F.interpolate(maps, size=inputs.shape[-2:], mode='bilinear', align_corners=False)

            # extract the landmark and existence mask, [N, num_landmarks, 2]
            visible = landmarks_full[:, :, 3] > 0.5
            points = landmarks_full[:, :, 1:3].unsqueeze(2).clone()

            points[:, :, :, 0] /= inputs.shape[-1]  # W
            points[:, :, :, 1] /= inputs.shape[-2]  # H
            assert points.min() > -1e-7 and points.max() < 1 + 1e-7
            points = points * 2 - 1

            pred_parts_loc_w_bg = F.grid_sample(part_name_mat_w_bg.float(), points, mode='nearest', align_corners=False)
            pred_parts_loc_w_bg = torch.argmax(pred_parts_loc_w_bg, dim=1).squeeze(2)
            pred_parts_loc_w_bg = pred_parts_loc_w_bg[visible]
            all_nmi_preds_w_bg.append(pred_parts_loc_w_bg.cpu().numpy())

            gt_parts_loc = torch.arange(landmarks_full.shape[1]).unsqueeze(0).repeat(landmarks_full.shape[0], 1).to(device, non_blocking=True)
            gt_parts_loc = gt_parts_loc[visible]
            all_nmi_gts.append(gt_parts_loc.cpu().numpy())

    nmi_preds = np.concatenate(all_nmi_preds_w_bg, axis=0)
    nmi_gts = np.concatenate(all_nmi_gts, axis=0)

    nmi = normalized_mutual_info_score(nmi_gts, nmi_preds) * 100
    ari = adjusted_rand_score(nmi_gts, nmi_preds) * 100
    return nmi, ari


def eval_kpr(net, fit_loader, eval_loader, nparts, num_landmarks, device=torch.device("cuda")):
    """
    Evaluate keypoint regression for given method
    Parameters
    ----------
    net: torch.nn.Module
        The trained net to evaluate
    fit_loader: torch.utils.data.DataLoader
        The dataset on which to train the keypoint regression
    eval_loader: torch.utils.data.DataLoader
        The dataset on which to evaluate the keypoint regression
    nparts: int
        Number of predicted parts
    num_landmarks: int
        Number of landmarks in the dataset
    device: torch.device
        The device to run the evaluation on
    Returns
    ----------
    kpr: float
        Keypoint regression between centroids of predicted parts and gt parts
    """
    # convert the assignment to centers for both splits
    print('Evaluating the model for the whole data split...')
    fit_centers, fit_annos, fit_masks, fit_active_centers, _, _, _, _ = \
        create_centers(fit_loader, net, nparts, num_landmarks, device)
    eval_centers, eval_annos, eval_masks, eval_active_centers, \
        gt_labels, pred_final, pred_landmarks, present_landmarks = \
        create_centers(eval_loader, net, nparts, num_landmarks, device)

    # fit the linear regressor with sklearn
    # normalized assignment center coordinates -> normalized landmark coordinate annotations
    print('=> fitting and evaluating the regressor')
    error = 0
    n_valid_samples = 0

    # different landmarks have different masks
    for i in range(num_landmarks):
        # get the valid indices for the current landmark
        fit_masks_np = fit_masks.cpu().numpy().astype(np.float64)
        eval_masks_np = eval_masks.cpu().numpy().astype(np.float64)
        fit_selection = (abs(fit_masks_np[:, i * 2]) > 1e-5)
        eval_selection = (abs(eval_masks_np[:, i * 2]) > 1e-5)

        fit_active_centers_select = fit_active_centers[fit_selection]
        eval_active_centers_select = eval_active_centers[eval_selection]

        # convert tensors to numpy (64 bit double)
        fit_centers_np = fit_centers.cpu().numpy().astype(np.float64)
        fit_annos_np = fit_annos.cpu().numpy().astype(np.float64)
        eval_centers_np = eval_centers.cpu().numpy().astype(np.float64)
        eval_annos_np = eval_annos.cpu().numpy().astype(np.float64)

        # select the current landmarks for both fit and eval set
        fit_annos_np = fit_annos_np[:, i * 2:i * 2 + 2]
        eval_annos_np = eval_annos_np[:, i * 2:i * 2 + 2]

        # remove invalid indices
        fit_centers_np = fit_centers_np[fit_selection]
        fit_annos_np = fit_annos_np[fit_selection]
        eval_centers_np = eval_centers_np[eval_selection]
        eval_annos_np = eval_annos_np[eval_selection]
        eval_data_size = eval_centers_np.shape[0]

        # data standardization
        scaler_centers = StandardScaler()
        scaler_landmarks = StandardScaler()

        # fit the StandardScaler with the fitting split
        scaler_centers.fit(fit_centers_np)
        scaler_landmarks.fit(fit_annos_np)

        # standardize the fitting split
        fit_centers_std = scaler_centers.transform(fit_centers_np)
        fit_annos_std = scaler_landmarks.transform(fit_annos_np)

        # Take only the active centers
        fit_centers_std = np.where(fit_active_centers_select.cpu().numpy(), fit_centers_std, 0)

        # define regressor without intercept and fit it
        regressor = LinearRegression(fit_intercept=False)
        regressor.fit(fit_centers_std, fit_annos_std)

        # standardize the centers on the evaluation split
        eval_centers_std = scaler_centers.transform(eval_centers_np)

        # Take only the active centers
        eval_centers_std = np.where(eval_active_centers_select.cpu().numpy(), eval_centers_std, 0)

        # regress the landmarks on the evaluation split
        eval_pred_std = regressor.predict(eval_centers_std)

        # unstandardize the prediction with StandardScaler for landmarks
        eval_pred = scaler_landmarks.inverse_transform(eval_pred_std)

        # calculate the error
        eval_pred = eval_pred.reshape((eval_data_size, 1, 2))
        eval_annos_np = eval_annos_np.reshape((eval_data_size, 1, 2))
        error += distance_l2(eval_pred, eval_annos_np) * eval_data_size
        n_valid_samples += eval_data_size

    error = error * 100 / n_valid_samples
    print('Mean L2 Distance on the test set is %.2f%%.' % error)
    print('Evaluation finished for model.')
    return error
