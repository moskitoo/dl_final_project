from data_processing_tools import *


dataset = BrightfieldMicroscopyDataset(train=True, validation=False)

train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

results = perform_segmentation_evaluation(train_dataloader, print_interval=20, save_images=True, target_dir='test', single_channel_target_dir='test_single_channel')