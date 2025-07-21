import pandas as pd
import os
import re

def load_tabular_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            # Assign house_id starting from 1, assuming image filenames start from 1
            data.append([i + 1, float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    df = pd.DataFrame(data, columns=['house_id', 'bedrooms', 'bathrooms', 'area', 'zipcode', 'price'])
    return df

def create_image_paths(base_image_dir):
    image_files = [f for f in os.listdir(base_image_dir) if f.endswith('.jpg')]
    image_paths_dict = {}
    for f in image_files:
        match = re.match(r'(\d+)_(\w+).jpg', f)
        if match:
            house_id = int(match.group(1))
            room_type = match.group(2)
            if house_id not in image_paths_dict:
                image_paths_dict[house_id] = {}
            image_paths_dict[house_id][room_type] = os.path.join(base_image_dir, f)
    return image_paths_dict

if __name__ == '__main__':
    data_filepath = 'Houses-dataset-master/Houses Dataset/HousesInfo.txt'
    base_image_dir = 'Houses-dataset-master/Houses Dataset'

    df = load_tabular_data(data_filepath)
    df.to_csv('housing_data.csv', index=False)
    print('Tabular data saved to housing_data.csv')

    image_data_dict = create_image_paths(base_image_dir)

    all_image_paths = []
    missing_images_count = 0
    for house_id in df['house_id']:
        house_images = image_data_dict.get(house_id, {})
        expected_types = ['bathroom', 'bedroom', 'frontal', 'kitchen']
        current_house_image_paths = {}
        for img_type in expected_types:
            if img_type in house_images:
                current_house_image_paths[img_type] = house_images[img_type]
            else:
                current_house_image_paths[img_type] = None
                missing_images_count += 1
        all_image_paths.append(current_house_image_paths)

    if missing_images_count > 0:
        print(f'WARNING: {missing_images_count} images are missing based on expected house_id and room_type combinations.')
    else:
        print('All expected image paths found.')

    print('\nSample image paths for first 5 houses:')
    for i in range(min(5, len(all_image_paths))):
        print(f'House {df["house_id"][i]}: {all_image_paths[i]}')


