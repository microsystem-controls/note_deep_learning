import pandas as pd
import numpy as np
import os
from PIL import Image
import re


def csvToTorch(country):
    csv_path = f'assets/csv/{country}.csv'
    df = pd.read_csv(csv_path)

    output_images_dir = f'assets/torch/{country}/imgs/'
    output_labels_path = f'assets/torch/{country}/labels.txt'

    os.makedirs(output_images_dir, exist_ok=True)

    labels_list = []

    def sanitize_filename(filename):
        # Replace invalid characters with underscores
        filename = re.sub(r'[<>:"/\\|?*, ]', '', filename)
        # Optionally, truncate the filename if it's too long (e.g., limit to 255 characters)
        return filename

    for (path, insertion), group in df.groupby(['path', 'insertion']):
        # Extract the label information
        is_genuine = group['isGenuine'].iloc[0]
        channel = group['channel'].iloc[0]
        
        # Convert data strings to numpy arrays
        resized_data = []
        for data_str in group['data']:
            # Convert the space-separated string to a numpy array
            data_array = np.fromstring(data_str, sep=' ')
            
            # Resize the array to a width of 280 using linear interpolation
            data_resized = np.interp(
                np.linspace(0, len(data_array) - 1, 280),
                np.arange(len(data_array)),
                data_array
            )
            
            resized_data.append(data_resized)
        
        # Stack the resized arrays vertically to form the data matrix
        if len(resized_data) > 10:
            if len(resized_data) % 10 != 0:
                print("unknown bug, you need to investigate", flush=True)
            else:
                # weird bug: sometimes resized_data is greater than 10 for frauds, maybe there was a duplicate note?
                resized_data = resized_data[0:10]
        data_matrix = np.vstack(resized_data)
        
        # Normalize each row to the range 0-255
        # Compute the min and range for each row
        row_min = data_matrix.min(axis=1, keepdims=True)
        row_ptp = data_matrix.ptp(axis=1, keepdims=True)  # Range of each row (max - min)

        # Avoid division by zero by setting zero ranges to one (or some other small value)
        row_ptp[row_ptp == 0] = 1

        # Apply normalization to each row
        data_matrix = 255 * (data_matrix - row_min) / row_ptp

        # Convert to uint8
        data_matrix = data_matrix.astype(np.uint8)
        # Normalize the data to the range 0-255 for image representation
        # data_matrix = (255 * (data_matrix - np.min(data_matrix)) / np.ptp(data_matrix)).astype(np.uint8)
        
        # Convert the numpy array to a PIL image
        img = Image.fromarray(data_matrix)
        
        # Incorporate the 'path' into the filename and sanitize it
        sanitized_path = sanitize_filename(f'{path}_{insertion}.jpg')
        
        # Save the image in the output directory
        img.save(os.path.join(output_images_dir, sanitized_path))
        
        # Append the label information to the list
        labels_list.append({
            'filename': sanitized_path,
            'isGenuine': is_genuine,
            'channel': channel
        })

    labels_df = pd.DataFrame(labels_list)
    labels_df.to_csv(output_labels_path, index=False)


if __name__ == "__main__":
    csvToTorch("Thailand")
    # csvToTorch("Australia")
