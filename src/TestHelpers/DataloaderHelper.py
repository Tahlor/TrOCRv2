from src.PositionEmbeddings.IAMDataset import IAMDataset
from src.PositionEmbeddings.IAMDatasetTwoLines import IAMDatasetTwoLines
from torch.utils.data import DataLoader
import pandas as pd

def _new_iam_df(label_directory):
    rows = []
    with open(label_directory, 'r') as label_file:
        for line in label_file:
            data = line.split('|')
            file_name = f'{data[0]}.jpg'
            label = ' '.join(data[1:]).replace('\n','')
            new_row = {'file_name': file_name, 'text': label}
            rows.append(new_row)
    df = pd.DataFrame(rows, columns=['file_name','text'])
    return df

def _get_new_train_eval_test_dfs(root_directory):
    train_df = _new_iam_df(root_directory + 'IAM/train_labels.txt')
    eval_df = _new_iam_df(root_directory + 'IAM/validation_labels.txt')
    test_df = _new_iam_df(root_directory + "IAM/test_labels.txt")
    return train_df, eval_df, test_df

def get_train_eval_test_dataloaders(processor, root_directory, batch_size, num_workers, use_double, num_images):
    train_df, eval_df, test_df = _get_new_train_eval_test_dfs(root_directory)

    train_dataset = None
    eval_dataset = None
    test_dataset = None
    if use_double:
        train_dataset = IAMDatasetTwoLines(root_dir=root_directory + 'IAM/AllImages/',
                            df=train_df,
                            processor=processor,
                            num_images = num_images)
        eval_dataset = IAMDatasetTwoLines(root_dir=root_directory + 'IAM/AllImages/',
                                df=eval_df,
                                processor=processor,
                            num_images = num_images)
        test_dataset = IAMDatasetTwoLines(root_dir=root_directory + 'IAM/AllImages/',
                                df=test_df,
                                processor=processor,
                            num_images = num_images)
    else:
        train_dataset = IAMDataset(root_dir=root_directory + 'IAM/AllImages/',
                            df=train_df,
                            processor=processor,
                            num_images = num_images)
        eval_dataset = IAMDataset(root_dir=root_directory + 'IAM/AllImages/',
                                df=eval_df,
                                processor=processor,
                            num_images = num_images)
        test_dataset = IAMDataset(root_dir=root_directory + 'IAM/AllImages/',
                                df=test_df,
                                processor=processor,
                            num_images = num_images)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, eval_dataloader, test_dataloader