import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import sys
# import config
from io import StringIO
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from torchinfo import summary
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {"Mayo 0": 0, "Mayo 1": 1, "Mayo 2": 2, "Mayo 3": 3}  # Assign labels

        # Collect all images with labels
        self.image_paths = []
        for class_name, label in self.classes.items():
            class_dir = os.path.join(root_dir, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Image formats
                    self.image_paths.append((os.path.join(class_dir, img_file), label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label  # Return image tensor and label

# # Step 1: Compute Mean and Std for Dataset
# def compute_mean_std(root_dir):
#     transform = transforms.Compose([transforms.ToTensor()])
#     dataset = datasets.ImageFolder(root=root_dir, transform=transform)
#     loader = DataLoader(dataset, batch_size=64, shuffle=False)

#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     num_batches = 0

#     for images, _ in loader:
#         num_batches += 1
#         for i in range(3):  # Loop over RGB channels
#             mean[i] += images[:, i, :, :].mean()
#             std[i] += images[:, i, :, :].std()

#     mean /= num_batches
#     std /= num_batches

#     return mean.tolist(), std.tolist()

class ROIExtractor:
    def __init__(self, num_classes=4):
        """
        Initialize ROI extractor

        Args:
            num_classes: Total number of classes in the classification problem
        """
        self.num_classes = num_classes

    def compute_distance_metric(self, prediction, k):
        """
        Compute the distance metric D_P^k(x) as defined in equation (6)

        Args:
            prediction: The predicted class P(x)
            k: The sub-CNN index

        Returns:
            Distance metric value
        """
        if k <= prediction:
            return 1.0 / (prediction - k + 1)
        else:  # k > prediction
            return 1.0 / (k - prediction)

    def extract_roi(self, ranking_outputs, cam_outputs, img_shape=None):
        """
        Extract Region of Interest (ROI) based on Algorithm 2 from the paper
        using pre-computed ranking outputs and CAMs

        Args:
            ranking_outputs: Binary classification outputs from RankingCNN [B, num_subcnns]
            cam_outputs: List of tuples (cam0, cam1) for each sub-CNN
            img_shape: Optional target shape for resizing CAMs (H, W)

        Returns:
            ROI tensor of shape [B, H, W]
        """
        # Get prediction (P(x)) for each image in batch
        # The prediction is determined by the number of values above 0.5 threshold
        predictions = torch.sum(ranking_outputs > 0.5, dim=1)

        batch_size = ranking_outputs.shape[0]
        num_subcnns = len(cam_outputs)
        device = ranking_outputs.device

        # Process CAMs - get cam0 and cam1 for each sub-CNN
        all_cam0 = []
        all_cam1 = []

        for i, (cam0, cam1) in enumerate(cam_outputs):
            # Apply ReLU to focus on positive contributions
            cam0_normalized = F.relu(cam0)
            cam1_normalized = F.relu(cam1)

            # Resize if needed
            if img_shape is not None and img_shape != cam0.shape[1:]:
                cam0_resized = F.interpolate(
                    cam0_normalized.unsqueeze(1),
                    size=img_shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

                cam1_resized = F.interpolate(
                    cam1_normalized.unsqueeze(1),
                    size=img_shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            else:
                cam0_resized = cam0_normalized
                cam1_resized = cam1_normalized

            all_cam0.append(cam0_resized)
            all_cam1.append(cam1_resized)

        # Get CAM shape
        h, w = all_cam0[0].shape[1:]

        # Initialize ROI tensor
        roi = torch.zeros(batch_size, h, w, device=device)

        # Process each image in the batch
        for b in range(batch_size):
            prediction = predictions[b].item()

            # Case 1: P(x) = 0 (normal eyes)
            if prediction == 0:
                roi[b] = all_cam0[0][b]  # C_0^1(x)

            # Case 2: P(x) = N-1 (severe glaucoma)
            elif prediction == self.num_classes - 1:
                roi[b] = all_cam1[num_subcnns - 1][b]  # C_1^(N-1)(x)

            # Case 3: P(x) in {1, 2, ..., N-2}
            else:
                # Initialize ROI with zeros
                temp_roi = torch.zeros_like(roi[b])

                # Apply equation (7) from the paper
                # For k <= P(x), use C_0^k(x)
                for k in range(1, prediction + 1):
                    # Compute D_P^k(x)
                    d_p_k = self.compute_distance_metric(prediction, k)
                    temp_roi += d_p_k * all_cam0[k-1][b]

                # For k > P(x), use C_1^k(x)
                for k in range(prediction + 1, self.num_classes):
                    # Compute D_P^k(x)
                    d_p_k = self.compute_distance_metric(prediction, k)
                    temp_roi += d_p_k * all_cam1[k-2][b]  # Adjust index since k starts from 1

                roi[b] = temp_roi

        return roi

# class CAM(nn.Module):
#     def __init__(self, num_classes=4, ranking_outputs, cam_outputs):
#         """
#         Initialize ROI extractor with a pre-trained Ranking CNN model

#         Args:
#             ranking_cnn_model: A trained RankingCNN model
#         """
#         # self.model = ranking_cnn_model
#         # self.model.eval()
#         super(CAM, self).__init__()
#         self.num_classes = num_classes

#     def compute_distance_metric(self, prediction, k):
#         """
#         Compute the distance metric D_P^k(x) as defined in equation (6)

#         Args:
#             prediction: The predicted class P(x)
#             k: The sub-CNN index

#         Returns:
#             Distance metric value
#         """
#         if k <= prediction:
#             return 1.0 / (prediction - k + 1)
#         else:  # k > prediction
#             return 1.0 / (k - prediction)

#     def generate_cams(self, img_tensor):
#         """
#         Generate Class Activation Maps (CAMs) for all sub-CNNs

#         Args:
#             img_tensor: Input image tensor of shape [B, 3, H, W]

#         Returns:
#             Tuple of (ranking_outputs, all_cam0, all_cam1)
#         """
#         # self.model.eval()
#         # with torch.no_grad():
#         #     # Forward pass through the model
#         #     ranking_outputs, cam_outputs = self.model(img_tensor)

#         # Extract CAMs
#         all_cam0 = []
#         all_cam1 = []

#         # For each sub-CNN
#         for i, (cam0, cam1) in enumerate(cam_outputs):
#             # Normalize CAMs
#             cam0_normalized = F.relu(cam0)
#             cam1_normalized = F.relu(cam1)

#             # Resize to match the input image size if needed
#             if img_tensor.shape[2:] != cam0.shape[1:]:
#                 cam0_resized = F.interpolate(
#                     cam0_normalized.unsqueeze(1),
#                     size=img_tensor.shape[2:],
#                     mode='bilinear',
#                     align_corners=False
#                 ).squeeze(1)

#                 cam1_resized = F.interpolate(
#                     cam1_normalized.unsqueeze(1),
#                     size=img_tensor.shape[2:],
#                     mode='bilinear',
#                     align_corners=False
#                 ).squeeze(1)
#             else:
#                 cam0_resized = cam0_normalized
#                 cam1_resized = cam1_normalized

#             all_cam0.append(cam0_resized)
#             all_cam1.append(cam1_resized)

#         return ranking_outputs, all_cam0, all_cam1

#     def extract_roi(self, img_tensor):
#         """
#         Extract Region of Interest (ROI) based on Algorithm 2 from the paper

#         Args:
#             img_tensor: Input image tensor of shape [B, 3, H, W]

#         Returns:
#             ROI tensor of shape [B, H, W]
#         """
#         # Get CAMs and predictions
#         ranking_outputs, all_cam0, all_cam1 = self.generate_cams(img_tensor)

#         # Get prediction (P(x)) for each image in batch
#         # The prediction is determined by the number of values above 0.5 threshold
#         predictions = torch.sum(ranking_outputs > 0.5, dim=1)

#         batch_size = img_tensor.shape[0]
#         num_classes = 3 + 1  # N
#         device = img_tensor.device

#         # Initialize ROI tensor
#         roi = torch.zeros(batch_size, img_tensor.shape[2], img_tensor.shape[3], device=device)

#         # Process each image in the batch
#         for b in range(batch_size):
#             prediction = predictions[b].item()

#             # Case 1: P(x) = 0 (normal eyes)
#             if prediction == 0:
#                 roi[b] = all_cam0[0][b]  # C_0^1(x)

#             # Case 2: P(x) = N-1 (severe glaucoma)
#             elif prediction == num_classes - 1:
#                 roi[b] = all_cam1[num_classes - 2][b]  # C_1^(N-1)(x)

#             # Case 3: P(x) in {1, 2, ..., N-2}
#             else:
#                 # Initialize ROI with zeros
#                 temp_roi = torch.zeros_like(roi[b])

#                 # Apply equation (7) from the paper
#                 # For k <= P(x), use C_0^k(x)
#                 for k in range(1, prediction + 1):
#                     # Compute D_P^k(x)
#                     d_p_k = self.compute_distance_metric(prediction, k)
#                     temp_roi += d_p_k * all_cam0[k-1][b]

#                 # For k > P(x), use C_1^k(x)
#                 for k in range(prediction + 1, num_classes):
#                     # Compute D_P^k(x)
#                     d_p_k = self.compute_distance_metric(prediction, k)
#                     temp_roi += d_p_k * all_cam1[k-2][b]  # Adjust index since k starts from 1

#                 roi[b] = temp_roi

#         return roi

class CAM(nn.Module):
    def __init__(self, num_classes=4, scale_factor=32):
        super(CAM, self).__init__()
        self.num_classes = num_classes
        self.scale_factor = scale_factor  # Scale factor for upsampling (16*32=512)

        # Dynamic upsampling layers
        self.upsample_layers = nn.ModuleList()
        current_scale = 1

        # Calculate number of upsampling stages needed
        target_scale = self.scale_factor
        while current_scale < target_scale:
            current_scale *= 2
            self.upsample_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(1, 1, kernel_size=3, padding=1),  # Refine features after upsampling
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, cam_features, ranking_outputs):
        # Apply ReLU to keep only positive activations
        # cam_features = F.relu(cam_features)

        # Global Average Pooling to get weights
        # cam_weights = cam_features.mean(dim=(2, 3), keepdim=True)

        # Apply weights to CAM
        # weighted_cam = cam_features * cam_weights
        weighted_cam = cam_features

        # print("weighted_cam", weighted_cam)

        # Compute the inter-class distance metric
        batch_size, num_subclasses, H, W = weighted_cam.shape
        roi_features = torch.zeros((batch_size, H, W), device=cam_features.device)

        # Iterate over each sub-CNN classifier
        for k in range(self.num_classes - 1):
            p_k = ranking_outputs[:, k]  # Binary output of k-th sub-CNN

            # Apply inter-class distance metric (Equation 6)
            distance_metric = torch.where(
                k <= ranking_outputs.argmax(dim=1),
                1.0 / (ranking_outputs.argmax(dim=1) - k + 1),  # k ≤ P(x)
                1.0 / (k - ranking_outputs.argmax(dim=1))  # k > P(x)
            )

            # Get CAMs for both sub-CNN binary classifiers
            cam_lower = weighted_cam[:, k, :, :] * (1 - ranking_outputs[:, k].view(-1, 1, 1))
            cam_higher = weighted_cam[:, k + 1, :, :] * ranking_outputs[:, k].view(-1, 1, 1)

            # Compute weighted region of interest (ROI)
            roi_features += distance_metric.view(-1, 1, 1) * (cam_lower + cam_higher)

        # Add channel dimension
        roi_features = roi_features.unsqueeze(1)  # Shape: [batch_size, 1, H, W]

        # Apply sequential upsampling if using learned upsampling
        if hasattr(self, 'upsample_layers') and len(self.upsample_layers) > 0:
            x = roi_features
            for layer in self.upsample_layers:
                x = layer(x)
            upsampled_roi = x
        else:
            # Alternatively, use direct interpolation to target size
            input_height, input_width = roi_features.shape[2], roi_features.shape[3]
            output_height = input_height * self.scale_factor
            output_width = input_width * self.scale_factor
            upsampled_roi = F.interpolate(
                roi_features,
                size=(output_height, output_width),
                mode='bilinear',
                align_corners=False
            )

        # print("ROI features shape is:", upsampled_roi.shape)
        return upsampled_roi

# class CAM(nn.Module):
#     def __init__(self, num_classes=4):
#         super(CAM, self).__init__()
#         self.num_classes = num_classes

#     def forward(self, cam_features, ranking_outputs):
#         cam_features = F.relu(cam_features)  # Apply ReLU to keep only positive activations
#         cam_weights = cam_features.mean(dim=(2, 3), keepdim=True)  # Global Average Pooling
#         # cam_weights = cam_features
#         # Apply weights to CAM
#         weighted_cam = cam_features * cam_weights

#         # Compute the inter-class distance metric (D_P^k(x))
#         batch_size, num_subclasses, H, W = weighted_cam.shape
#         roi_features = torch.zeros((batch_size, H, W), device=cam_features.device)  # Initialize ROI

#         # Iterate over each sub-CNN classifier
#         for k in range(self.num_classes - 1):
#             p_k = ranking_outputs[:, k]  # Binary output of k-th sub-CNN

#             # Apply inter-class distance metric (Equation 6)
#             distance_metric = torch.where(
#                 k <= ranking_outputs.argmax(dim=1),
#                 1.0 / (ranking_outputs.argmax(dim=1) - k + 1),  # k ≤ P(x)
#                 1.0 / (k - ranking_outputs.argmax(dim=1))  # k > P(x)
#             )

#             # Get CAMs for both sub-CNN binary classifiers
#             cam_lower = weighted_cam[:, k, :, :] * (1 - ranking_outputs[:, k].view(-1, 1, 1))  # C_0[k]
#             cam_higher = weighted_cam[:, k + 1, :, :] * ranking_outputs[:, k].view(-1, 1, 1)   # C_1[k]

#             # Compute weighted region of interest (ROI)
#             roi_features += distance_metric.view(-1, 1, 1) * (cam_lower + cam_higher)
#         print("ROI features shape is:", np.shape(roi_features))
#         return roi_features.unsqueeze(1)  # Add channel dimension

# Assuming this is the RankingCNN class (not fully shown in original code)
class RankingCNN(nn.Module):
    def __init__(self, num_classes=4, backbone='densenet121'):
        super(RankingCNN, self).__init__()
        self.num_subcnns = num_classes - 1

        # Create separate DenseNet backbones for each sub-CNN
        self.subcnn_backbones = nn.ModuleList()
        self.subcnn_projections = nn.ModuleList()
        self.subcnn_classifiers = nn.ModuleList()

        for _ in range(self.num_subcnns):
            # Create a separate DenseNet for each sub-CNN
            if backbone == 'densenet121':
                densenet = models.densenet121(pretrained = True)  # Note: channels last format in TF/Keras
                # Remove the classifier layer
                feature_extractor = nn.Sequential(*list(densenet.children())[:-1])
                # print("feature_extractor shape is:", feature_extractor)

                # summary_str = StringIO()
                # sys.stdout = summary_str  # Redirect stdout to capture summary
                # summary(feature_extractor, input_size=(4, 3, 512, 512))
                # sys.stdout = sys.__stdout__  # Reset stdout

                # full_output = "\nModel Summary:\n" + summary_str.getvalue()

                # # Save the information to a file
                # with open('training_info.txt', 'w') as f:
                #     f.write(full_output)

                self.subcnn_backbones.append(feature_extractor)
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")

            self.subcnn_classifiers.append(nn.Linear(1024, 2))  # Binary classification

            # Add projections for binary classification
            self.subcnn_projections.append(
                nn.Sequential(
                    nn.Linear(1024, 512),  # 1024 is the output dim of DenseNet121
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, 2),  # Binary classification with 2 classes
                )
            )

        # Flag to control whether to use global average pooling (should always be True based on the figure)
        self.use_global_pooling = True

    def forward(self, x):
        # Process input through each sub-CNN independently
        ranking_outputs = []
        features_list = []
        cam_outputs = []

        for i in range(self.num_subcnns):
            # Pass input through each sub-CNN's backbone
            features = self.subcnn_backbones[i](x)
            features_list.append(features)

            # Apply global average pooling
            # if self.use_global_pooling:
            # Global average pooling to get a 1024-dim vector
            pooled_features = torch.mean(features, dim=(2, 3))

            # Project to binary classification output
            logits = self.subcnn_projections[i](pooled_features)
            # print("logits", logits)
            ranking_output = torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)  # Extract probability for class 1
            # print("ranking_output_softmax", torch.softmax(logits, dim=1))
            # print("ranking_output_softmax", ranking_output)
            # print("sigmoid", torch.sigmoid(logits))
            # else:
            #     # This should not be used based on the figure, but keeping it for compatibility
            #     pooled_features = torch.mean(features, dim=(2, 3))

            #     logits = self.subcnn_projections[i](pooled_features)
            #     ranking_output = torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)
            #     # ranking_output = logits[:, 1].unsqueeze(1)
            ranking_outputs.append(ranking_output)

            # Compute CAM for both classes
            weights = self.subcnn_classifiers[i].weight  # Shape: [2, 1024]

            # Initialize CAMs with zeros
            batch_size = features.size(0)
            H, W = features.size(2), features.size(3)
            cam0 = torch.zeros(batch_size, H, W).to(features.device)
            cam1 = torch.zeros(batch_size, H, W).to(features.device)

            # Compute CAM by weighted sum of feature maps
            for b in range(batch_size):
                for c in range(1024):  # For each channel in features
                    cam0[b] += weights[0, c] * features[b, c]
                    cam1[b] += weights[1, c] * features[b, c]

            # ReLU to focus on features that have a positive influence
            cam0 = F.relu(cam0)
            cam1 = F.relu(cam1)

            # Add to cam_outputs
            cam_outputs.append((cam0, cam1))

        # Concatenate ranking outputs
        ranking_outputs = torch.cat(ranking_outputs, dim=1)

        # Return the first set of features for compatibility with the rest of the pipeline
        return ranking_outputs, cam_outputs, features_list

class FinalClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(FinalClassifier, self).__init__()
        # Process each input tensor through separate initial convolutions
        self.conv1_list = nn.ModuleList([
            nn.Conv2d(1024, 512, kernel_size=3, padding=1) for _ in range(3)
        ])
        self.bn1_list = nn.ModuleList([
            nn.BatchNorm2d(512) for _ in range(3)
        ])

        # Combine features and then process with a single second conv layer
        self.conv2 = nn.Conv2d(512*3, 256, kernel_size=3, padding=1)  # Concatenated features
        self.bn2 = nn.BatchNorm2d(256)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, images_list):
        # Process each tensor separately through first conv block
        features_list = []
        for i, images in enumerate(images_list):
            x = self.conv1_list[i](images)
            x = self.bn1_list[i](x)
            x = nn.functional.relu(x)
            features_list.append(x)

        # Concatenate features along the channel dimension
        x = torch.cat(features_list, dim=1)  # [batch_size, 512*3, 16, 16]

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)

        # Global average pooling
        x = torch.mean(x, dim=(2, 3))  # [batch_size, 256]

        # Classification
        x = self.classifier(x)
        return x

# # Final Classifier model using ROI features
# class FinalClassifier(nn.Module):
#     def __init__(self, num_classes=4):
#         super(FinalClassifier, self).__init__()
#         # First conv layer: specify out_channels
#         self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)  # [16, 1024, 16, 16] -> [16, 512, 16, 16]
#         self.bn1 = nn.BatchNorm2d(512)

#         # Second conv layer: specify in_channels and out_channels
#         self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # [16, 512, 16, 16] -> [16, 256, 16, 16]
#         self.bn2 = nn.BatchNorm2d(256)

#         # Global average pooling followed by classification head
#         self.classifier = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes)
#         )


#         self.regressor = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )

#         # Learnable polynomial coefficients
#         self.poly_a = nn.Parameter(torch.tensor(5.0))
#         self.poly_b = nn.Parameter(torch.tensor(-6.0))
#         self.poly_c = nn.Parameter(torch.tensor(4.0))
#         self.poly_d = nn.Parameter(torch.tensor(0.0))

#     def forward(self, images):
#         # First conv block
#         x = self.conv1(images)
#         x = self.bn1(x)
#         x = nn.functional.relu(x)
#         # Second conv block
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = nn.functional.relu(x)
#         # Global average pooling
#         x = torch.mean(x, dim=(2, 3))
#         # Output continuous value
#         # continuous_value = self.regressor(x)
#         # # Scale from [0,1] to [0, num_classes-1]
#         # # Apply learnable polynomial transformation
#         # polynomial_value = (self.poly_a * continuous_value**3 +
#         #                    self.poly_b * continuous_value**2 +
#         #                    self.poly_c * continuous_value +
#         #                    self.poly_d)

#         # # Clip to valid range
#         # scaled_value = torch.clamp(polynomial_value, 0.0, 3.0)

#         return self.classifier(x)


    # def forward(self, images):
    #     # First conv block
    #     x = self.conv1(images)  # input is [16, 1024, 16, 16]
    #     x = self.bn1(x)
    #     x = nn.functional.relu(x)
    #     # Second conv block
    #     x = self.conv2(x)
    #     x = self.bn2(x)
    #     x = nn.functional.relu(x)
    #     # Global average pooling
    #     x = torch.mean(x, dim=(2, 3))  # [16, 256, 16, 16] -> [16, 256]
    #     x = self.classifier(x)
    #     return x

# Phase 1: Train only the Ranking CNN
def train_ranking_cnn(ranking_cnn, train_loader, val_loader, num_epochs=2, lr=0.0001, device='cuda'):
    # Ensure global pooling is enabled for Phase 1
    ranking_cnn.use_global_pooling = True

    # Move model to device
    ranking_cnn = ranking_cnn.to(device)

    # Optimizer
    optimizer = optim.Adam(ranking_cnn.parameters(), lr=lr, weight_decay = 1e-4)

    # Loss function
    criterion = nn.BCELoss()  # Binary Cross Entropy for ranking outputs

    print("=== Phase 1: Training Ranking CNN with global pooling enabled ===")

    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        ranking_cnn.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Convert labels to binary ranking targets
            ranking_targets = []
            for label in labels:
                # For each label, create binary targets for N-1 classifiers
                binary_targets = torch.zeros(ranking_cnn.num_subcnns, device=device)
                for k in range(ranking_cnn.num_subcnns):
                    if label > k:                             
                        binary_targets[k] = 1.0
                ranking_targets.append(binary_targets)
            ranking_targets = torch.stack(ranking_targets)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            ranking_outputs, _, _ = ranking_cnn(inputs)

            # Calculate loss

            # print("ranking_outputs", ranking_outputs)
            # print("ranking_targets", ranking_targets)

            loss = criterion(ranking_outputs, ranking_targets) 

            # Backward + optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()

            # Calculate accuracy (predicted class = number of positive binary outputs)
            predicted_class = (ranking_outputs > 0.5).sum(dim=1)   
            total += labels.size(0)
            correct += (predicted_class == labels).sum().item()

            # print("predicted_class", predicted_class)
            # print("labels", labels)
            # print("(predicted_class == labels).sum().item()", (predicted_class == labels).sum().item())

        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {running_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {100 * correct / total:.2f}%')

        # Validation
        ranking_cnn.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Convert labels to binary ranking targets
                ranking_targets = []
                for label in labels:
                    binary_targets = torch.zeros(ranking_cnn.num_subcnns, device=device)
                    for k in range(ranking_cnn.num_subcnns):
                        if label > k:
                            binary_targets[k] = 1.0
                    ranking_targets.append(binary_targets)
                ranking_targets = torch.stack(ranking_targets)

                # Forward pass
                ranking_outputs, _, _ = ranking_cnn(inputs)

                # Calculate loss
                loss = criterion(ranking_outputs, ranking_targets)
                val_loss += loss.item()

                # Calculate accuracy
                predicted_class = (ranking_outputs > 0.5).sum(dim=1)
                val_total += labels.size(0)
                val_correct += (predicted_class == labels).sum().item()

        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        print(f'Validation Accuracy: {100 * val_correct / val_total:.2f}%')
        print('-' * 50)

    return ranking_cnn

import torch
import torch.nn as nn

class CDWCE(nn.Module):
    def __init__(self, alpha=1.0):
        super(CDWCE, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        """
        y_pred: Tensor of shape (batch_size, num_classes) - predicted probabilities for each class
        y_true: Tensor of shape (batch_size) - true class indices (single integer per sample)
        """
        batch_size, num_classes = y_pred.size()

        # Ensure predictions are in the range (0, 1) by applying sigmoid to avoid log(0)
        y_pred = torch.sigmoid(y_pred)  # Alternatively, you can use softmax if multi-class classification

        loss = 0.0
        for i in range(num_classes):
            # Calculate the distance factor |i - c|^alpha
            distance_factor = torch.abs(i - y_true) ** self.alpha

            # Log(1 - y_pred) for each class
            log_term = torch.log(1 - y_pred[:, i] + 1e-6)  # Small epsilon to avoid log(0)

            # Add weighted loss for each class
            loss -= (log_term * distance_factor).sum() / batch_size

        return loss


# Phase 2: Continue training the Ranking CNN while also training the Final Classifier
def train_combined_model(ranking_cnn, cam_module, final_classifier, train_loader, val_loader,
                         num_epochs=2, lr=0.001, device='cuda'):
    # Disable global pooling for Phase 2
    ranking_cnn.use_global_pooling = False

    # Move models to device
    ranking_cnn = ranking_cnn.to(device)
    cam_module = cam_module.to(device)
    final_classifier = final_classifier.to(device)

    # Combined optimizer for both models
    optimizer = optim.Adam(list(ranking_cnn.parameters()) + list(final_classifier.parameters()), lr=lr, weight_decay = 1e-4)

    # Loss functions
    ranking_criterion = nn.BCELoss()  # Binary Cross Entropy for ranking outputs
    classifier_criterion = nn.CrossEntropyLoss()  # Cross Entropy for final classification

    print("=== Phase 2: Combined Training of Ranking CNN (without global pooling) and Final Classifier ===")

    # Training loop
    for epoch in range(num_epochs):
        # Set models to training mode
        ranking_cnn.train()
        final_classifier.train()

        running_ranking_loss = 0.0
        running_classifier_loss = 0.0
        correct = 0
        total = 0

        for input_images, labels in train_loader:
            input_images, labels = input_images.to(device), labels.to(device)

            # Convert labels to binary ranking targets
            ranking_targets = []
            for label in labels:
                # For each label, create binary targets for N-1 classifiers
                binary_targets = torch.zeros(ranking_cnn.num_subcnns, device=device)
                for k in range(ranking_cnn.num_subcnns):
                    if label > k:
                        binary_targets[k] = 1.0
                ranking_targets.append(binary_targets)
            ranking_targets = torch.stack(ranking_targets)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Step 1: Forward pass through Ranking CNN
            ranking_outputs, feature_maps, _ = ranking_cnn(input_images)
            roi_extractor = ROIExtractor(num_classes=4)  # Adjust based on your model (e.g., 4 classes)

            # Extract ROI
            roi_features = roi_extractor.extract_roi(
                ranking_outputs,
                feature_maps,
                img_shape=[512, 512]  # Set to None if CAMs are already correct size
            )
            roi_features = roi_features.unsqueeze(1)

            # print("feature_maps.shape", feature_maps.shape)
            
            # print("ranking_outputs", ranking_outputs)
            # print("ranking_targets", ranking_targets)

            # Step 2: Calculate ranking loss

            # if ranking_outputs.shape[1] != ranking_targets.shape[1]:
            # # Apply dimensionality reduction if needed
            #     ranking_outputs = ranking_outputs[:, :ranking_targets.shape[1]]
            # ranking_outputs = torch.sigmoid(ranking_outputs)
            ranking_loss = ranking_criterion(ranking_outputs, ranking_targets)

            # Step 3: Get ROI using CAM
            # roi_features = cam_module(feature_maps, ranking_outputs)
            # print("Feature map shape is: ", np.shape(feature_maps))
            # print("Input image shape is:", np.shape(input_images))
            # Step 4: Forward pass through Final Classifier with features and ROI

            # print("roi_features.shape", roi_features.shape)
            # print("input_images.shape", input_images.shape)

            input_to_final_classifier = torch.cat([input_images, roi_features], dim=1)
            channel_converter = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1).to(device)
            input_to_final_classifier = channel_converter(input_to_final_classifier)  # input: [16, 4, 512, 512], output: [16, 3, 512, 512]
            # print("input_to_final_classifier", input_to_final_classifier.shape)
            # rgb = input_to_final_classifier[:, :3, :, :]
            # alpha = input_to_final_classifier[:, 3:4, :, :]

            # # Apply alpha blending with white background
            # white_background = torch.ones_like(rgb)
            # input_to_final_classifier = alpha * rgb + (1 - alpha) * white_background

            _, _, input_to_final_classifier = ranking_cnn(input_to_final_classifier)
            # print("input_to_final_classifier later", input_to_final_classifier[0].shape)

            classifier_outputs = final_classifier(input_to_final_classifier)   # replace feature_maps with input

            # Step 5: Compute classification loss
            classifier_loss = classifier_criterion(classifier_outputs, labels)

            classifier_outputs = torch.softmax(classifier_outputs, dim=1)

            # Total loss is a combination of ranking and classification losses
            total_loss = ranking_loss + classifier_loss

            # Step 6: Backward and optimize
            total_loss.backward()
            optimizer.step()

            # Statistics
            running_ranking_loss += ranking_loss.item()
            running_classifier_loss += classifier_loss.item()
            
            # print("classifier_outputs", classifier_outputs)
            _, predicted = torch.max(classifier_outputs, dim = 1)
            # predicted = predicted.T
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # print("classifier_outputs", classifier_outputs)
            # print("predicted", predicted)
            # print("labels", labels)
            # print("correct", correct)

        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Ranking Loss: {running_ranking_loss/len(train_loader):.4f}')
        print(f'Classifier Loss: {running_classifier_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {100 * correct / total:.2f}%')

        # Validation
        ranking_cnn.eval()
        final_classifier.eval()
        val_ranking_loss = 0.0
        val_classifier_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for input_images, labels in val_loader:
                input_images, labels = input_images.to(device), labels.to(device)

                # Convert labels to binary ranking targets
                ranking_targets = []
                for label in labels:
                    binary_targets = torch.zeros(ranking_cnn.num_subcnns, device=device)
                    for k in range(ranking_cnn.num_subcnns):
                        if label > k:
                            binary_targets[k] = 1.0
                    ranking_targets.append(binary_targets)
                ranking_targets = torch.stack(ranking_targets)

                # Forward pass through the full pipeline
                ranking_outputs, feature_maps, _ = ranking_cnn(input_images)
                # if ranking_outputs.shape[1] != ranking_targets.shape[1]:
                # # Apply dimensionality reduction if needed
                #   ranking_outputs = ranking_outputs[:, :ranking_targets.shape[1]]
                # ranking_outputs = torch.sigmoid(ranking_outputs)
                ranking_loss = ranking_criterion(ranking_outputs, ranking_targets)

                # roi_features = cam_module(feature_maps, ranking_outputs)

                roi_extractor = ROIExtractor(num_classes=4)  # Adjust based on your model (e.g., 4 classes)

                # Extract ROI
                roi_features = roi_extractor.extract_roi(
                    ranking_outputs,
                    feature_maps,
                    img_shape=[512, 512]  # Set to None if CAMs are already correct size
                )
                roi_features = roi_features.unsqueeze(1)

                input_to_final_classifier = torch.cat([input_images, roi_features], dim=1)
                channel_converter = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1).to(device)
                input_to_final_classifier = channel_converter(input_to_final_classifier)  # input: [16, 4, 512, 512], output: [16, 3, 512, 512]

                # rgb = input_to_final_classifier[:, :3, :, :]
                # alpha = input_to_final_classifier[:, 3:4, :, :]

                # # Apply alpha blending with white background
                # white_background = torch.ones_like(rgb)
                # input_to_final_classifier = alpha * rgb + (1 - alpha) * white_background

                _, _, input_to_final_classifier = ranking_cnn(input_to_final_classifier)

                classifier_outputs = final_classifier(input_to_final_classifier)   # replace feature_maps with input

                classifier_loss = classifier_criterion(classifier_outputs, labels)
                classifier_outputs = torch.softmax(classifier_outputs, dim=1)
                

                val_ranking_loss += ranking_loss.item()
                val_classifier_loss += classifier_loss.item()

                # print("val classifier outputs", classifier_outputs)

                _, predicted = torch.max(classifier_outputs, dim = 1)

                # print("predicted", predicted)
                # print("predicted.T", predicted.T)
                # print("labels", labels)
                val_total += labels.size(0)
                # print("predicted.eq(labels)", predicted.eq(labels))
                # print("predicted.eq(labels).sum()", predicted.eq(labels).sum())
                # print("predicted.eq(labels).sum().item()", predicted.eq(labels).sum().item())
                val_correct += predicted.eq(labels).sum().item()
                # sum = 0
                # for i in range(0, 4):
                #     if predicted[i] == labels[i]:
                #         sum+=1
                # val_correct += sum        

        print(f'Validation Ranking Loss: {val_ranking_loss/len(val_loader):.4f}')
        print(f'Validation Classifier Loss: {val_classifier_loss/len(val_loader):.4f}')
        print(f'Validation Accuracy: {100 * val_correct / val_total:.2f}%')
        print('-' * 50)

    return ranking_cnn, final_classifier

# Full training pipeline with two phases
def train_full_pipeline(ranking_cnn, cam_module, final_classifier, train_loader, val_loader,
                       num_epochs=20, lr=0.0001, device='cuda'):
    # Phase 1: Train Ranking CNN with global pooling
    ranking_cnn = train_ranking_cnn(ranking_cnn, train_loader, val_loader, num_epochs, lr, device)

    # Phase 2: Continue training Ranking CNN (without global pooling) and train Final Classifier
    ranking_cnn, final_classifier = train_combined_model(
        ranking_cnn, cam_module, final_classifier,
        train_loader, val_loader, num_epochs, lr, device
    )

    return ranking_cnn, cam_module, final_classifier

# Inference function
def inference(ranking_cnn, cam_module, final_classifier, image, device='cuda'):
    # Prepare image
    if not isinstance(image, torch.Tensor):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)

    image = image.to(device)

    # Set models to evaluation mode
    ranking_cnn.eval()
    final_classifier.eval()

    # For inference, we want to use the global pooling
    ranking_cnn.use_global_pooling = False

    with torch.no_grad():
        # Step 1: Forward pass through Ranking CNN
        ranking_outputs, feature_maps = ranking_cnn(image)

        # Step 2: Get ROI using CAM
        roi_features = cam_module(feature_maps, ranking_outputs)

        # Step 3: Forward pass through Final Classifier
        outputs = final_classifier(feature_maps, roi_features)
        # print("outputs are: ", outputs)
        # Get predicted class
        predicted = outputs.item()

        return predicted.item(), outputs, roi_features

# Example usage
if __name__ == "__main__":
    # config = vars(parse_args())
    #  Data Augmentation Pipeline
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
        # transforms.RandomRotation(degrees=15),   # Randomly rotate ±15 degrees
        transforms.Resize((512, 512)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust color
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    #  No Augmentation for Validation/Test
    test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(root_dir="train_and_validation_sets", transform=train_transform)

    # Compute split sizes
    total_size = len(dataset)
    train_size = int(0.65 * total_size)
    val_size = int(0.20 * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Initialize models
    num_classes = 4
    ranking_cnn = RankingCNN(num_classes=num_classes)
    cam_module = CAM(num_classes=num_classes)
    final_classifier = FinalClassifier(num_classes=num_classes)

    # Assuming you have train_loader and val_loader set up
    # train_loader = ...
    # val_loader = ...

    # Train the models using the two-phase approach
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ranking_cnn = ranking_cnn.to(device)
    ranking_cnn, cam_module, final_classifier = train_full_pipeline(
        ranking_cnn, cam_module, final_classifier,
        train_loader, val_loader,
        num_epochs=10,
        device=device
    )

    # Save the trained models
    torch.save(ranking_cnn.state_dict(), 'ranking_cnn.pth')
    torch.save(final_classifier.state_dict(), 'final_classifier.pth')
    torch.cuda.empty_cache()