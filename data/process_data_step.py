from geopy.geocoders import Nominatim
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.neighbors import BallTree
import pandas as pd
# from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
from openai import OpenAI
from tqdm import tqdm
import random
from os.path import join
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from transformers import pipeline
import torch
import json
import jsonlines
from tqdm import tqdm
import time
from transformers import AutoTokenizer

target_dataset = 'TKY_deepseek'  #['TKY_Qwen','CA_Qwen','NYC_Qwen']
source_pth = f'./raw/{target_dataset}'
##################################### Obtain train completion format ############################
def remap(df: pd.DataFrame, n_users, n_pois):
    uid_dict = dict(zip(pd.unique(df['uid']), range(n_users)))
    poi_dict = dict(zip(pd.unique(df['poi']), range(n_pois)))
    df['uid'] = df['uid'].map(uid_dict)
    df['poi'] = df['poi'].map(poi_dict)
    return df, uid_dict, poi_dict

random.seed(1234)
target_dataset = 'TKY' # ['Foursquare', 'Gowalla'] "Foursquare" denotes the Singapore dataset
# target_dataset = 'Gowalla' # ['Foursquare', 'Gowalla']
source_pth = f'./raw/{target_dataset}'

trn_df = pd.read_csv(join(source_pth, 'train_sample.csv'),encoding='unicode_escape')
val_df = pd.read_csv(join(source_pth, 'val_sample.csv'), encoding='unicode_escape')
tst_df = pd.read_csv(join(source_pth, 'test_sample.csv'),encoding='unicode_escape')
# sam_df = pd.read_csv(join(source_pth, 'sample.csv'),encoding='unicode_escape')

review_df = pd.concat((trn_df, val_df, tst_df))
# review_df = review_df[['UserId','PoiId','Latitude', 'Longitude','UTCTimeOffset']]
# review_df.columns = ['uid', 'poi', 'lat', 'lon', 'time']
column_rename_map = {
    'UserId': 'uid',
    'PoiId': 'poi',  # Rename PoiId to poi
    'Latitude': 'lat',  # Rename Latitude to lat
    'Longitude': 'lon',  # Rename Longitude to lon
    'UTCTimeOffset': 'time'
    # Add other columns that need to be renamed, e.g., 'UserId': 'uid'
}
review_df.rename(columns=column_rename_map, inplace=True)
# print(review_df)

n_user, n_poi = pd.unique(review_df['uid']).shape[0], pd.unique(review_df['poi']).shape[0]
review_df, uid_dic, poi_dic = remap(review_df, n_user, n_poi)
review_df['time'] = (pd.to_datetime(review_df['time']).astype('int64') // 1e9).astype('int64')
review_df.sort_values(by=['uid','time'], inplace=True)

#Generate data
trn_df1, val_df1, tst_df1 = [], [], []
trn_set, val_set, tst_set = [], [], []
pivot_values = []

for uid, line in review_df.groupby('uid'):
    val_piv, test_piv = np.quantile(line['time'], [0.8, 0.9])
    pivot_values.append({'uid': uid, 'val_piv': val_piv, 'test_piv': test_piv})
    trn_df1.append(line[line['time'] < val_piv])
    val_df1.append(line[(line['time'] >= val_piv) & (line['time'] < test_piv)])
    tst_df1.append(line[line['time'] >= test_piv])

trn_df, val_df, tst_df = pd.concat(trn_df1), pd.concat(val_df1), pd.concat(tst_df1)
pivot_df = pd.DataFrame(pivot_values)
pivot_df.sort_values(by='uid', inplace=True)
pivot_df.to_csv(join(source_pth,'time_ca.csv'), index=False)

trn_df.to_csv(join(source_pth,'train_data_media.csv'), index=False)
val_df.to_csv(join(source_pth, 'val_data_media.csv'), index=False)
tst_df.to_csv(join(source_pth, 'tst_data_media.csv'), index=False)

##################################### Obtain train completion format ############################

########################## 1. CA: Process data divided by time into format with trajectory_id ##################
from datetime import datetime
def convert_utc_to_timestamp(utc_string):
    """
    Convert UTC time string to Unix timestamp
    Example: '2010-08-19 00:49:02' -> 1282179742
    """
    if pd.isna(utc_string) or not isinstance(utc_string, str):
        return np.nan

    try:
        dt = datetime.strptime(utc_string, '%Y-%m-%d %H:%M:%S')
        return int(dt.timestamp())
    except ValueError as e:
        print(f"Time format error: {utc_string}")
        return np.nan

random.seed(1234)
sam_df = pd.read_csv(join(source_pth, 'train_data_media.csv'),encoding='unicode_escape')
column_rename_map = {
    'pseudo_session_trajectory_id': 'trajectory_id',
    'uid': 'UserId',
    'poi': 'PoiId',
    'lat': 'Latitude',
    'lon': 'Longitude',
    'time': 'UTCTimeOffset'}

sam_df.rename(columns=column_rename_map, inplace=True)
# sam_df['UTCTimeOffset'] = sam_df['UTCTimeOffset'].apply(convert_utc_to_timestamp)

# #dataset=='CA'
# sam_df = sam_df[['check_ins_id','UTCTimeOffset','UTCTimeOffsetEpoch','Time','UserId',
#                  'Latitude','Longitude','PoiId','PoiCategoryId','PoiCategoryName','trajectory_id']]
#else:
sam_df = sam_df[['check_ins_id','UTCTimeOffset','UTCTimeOffsetEpoch','UserId',
                 'Latitude','Longitude','PoiId','PoiCategoryId','PoiCategoryName','trajectory_id']]

sam_df.to_csv(join(source_pth,'1_train_data_media_trajid_all1.csv'), index=False)
random.seed(1234)

########################## 1. Process data divided by time into format with trajectory_id ##################

####################### 2. Generate DetailedLocation (_location) #################################
# Input and output file names
input_file = "raw/TKY_deepseek/1_train_data_media_trajid_all.csv"
output_file = "raw/TKY_deepseek/2_ca_location_en.csv"

# Read CSV file
data = pd.read_csv(input_file)

# Check if required columns exist
if not {'Latitude', 'Longitude', 'PoiCategoryName'}.issubset(data.columns):
    raise ValueError("CSV file missing required columns: Latitude, Longitude, PoiCategoryName")

# Initialize geocoder
geolocator = Nominatim(user_agent="HXY")

def get_address(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=15,language = "en")
        time.sleep(1)
        if location and location.address:
            # Split address into components
            address = location.raw.get("address", {})
            # city = address.get("city")
            district = address.get("suburb", address.get("county"))
            road = address.get("road")
            house_number = address.get("house_number")
            components = [component for component in [district, road, f"No.{house_number}" if house_number else None] if component]
            return " ".join(components) if components else "Unknown"
        return "Unknown"
    except GeocoderTimedOut:
        return "Timeout"

# Initialize cache
address_cache = {}

# Add a new column to store detailed address
def get_detailed_location(row):
    poi_id = row['PoiId']
    if poi_id in address_cache:
        return address_cache[poi_id]

    # Calculate new address
    detailed_location = get_address(row['Latitude'], row['Longitude'])
    detailed_location_with_category = f"{detailed_location}({row['PoiCategoryName']})"

    # Cache result
    address_cache[poi_id] = detailed_location_with_category
    return detailed_location_with_category

# Apply function to generate detailed address column
data['DetailedLocation'] = data.apply(get_detailed_location, axis=1)

# Output to new CSV file
data.to_csv(output_file, index=False, encoding="utf-8")

print(f"Conversion complete, output file: {output_file}")
########################### 2. Generate DetailedLocation (_location) #################################

########################### 3. Remove trajectories with count below a threshold, and remove user data with fewer than 10 trajectories ##############################
import pandas as pd

input_file = "2_tky_location_en.csv"
output_file = "3_train_trajid_poi10.csv"
deleted_file = "3_1_deleted_traj_less_poi10.csv"
# input_file = "2_nyc_location.csv"
# output_file = "3_train_trajid_poi10.csv"
# deleted_file = "3_1_deleted_traj_less_poi10.csv"

# Read original data
df = pd.read_csv(join(source_pth, input_file))

# 1. First, count the occurrences of each trajectory_id
trajectory_counts = df['trajectory_id'].value_counts()

# 2. Get trajectory_ids with counts < 10, preserving original order
keep_small_df = df[df['trajectory_id'].isin(
    trajectory_counts[trajectory_counts < 10].index)]

# 3. Process trajectory_ids with counts >= 10
large_trajectories = trajectory_counts[trajectory_counts >= 10].index
deleted_data = pd.DataFrame()  # To store data to be deleted
keep_large_df = pd.DataFrame()  # To store data with counts >= 10 that does not meet conditions

for traj_id in large_trajectories:
    # Get subset for current trajectory_id
    traj_df = df[df['trajectory_id'] == traj_id].copy()

    # Convert UTCTimeOffset to numeric type
    traj_df['UTCTimeOffset'] = pd.to_numeric(traj_df['UTCTimeOffset'])

    # Calculate time span (days)
    min_time = traj_df['UTCTimeOffset'].min()
    max_time = traj_df['UTCTimeOffset'].max()
    days = (max_time - min_time) / (24 * 3600)  # Convert to days

    # Calculate counts/days
    counts = len(traj_df)
    rate = counts / days if days > 0 else float('inf')  # Prevent division by zero

    # Determine whether to delete
    if rate >= 10:
        deleted_data = pd.concat([deleted_data, traj_df])
    else:
        keep_large_df = pd.concat([keep_large_df, traj_df])

# 4. Merge preliminary retained data (counts < 10 and counts >= 10 but rate < 10)
preliminary_df = pd.concat([keep_small_df, keep_large_df])
final_df = preliminary_df.sort_values(['UserId', 'UTCTimeOffset'])
# Save final result
final_df.to_csv(join(source_pth, output_file), index=False)

# If there is deleted data, save to another CSV file
if not deleted_data.empty:
    deleted_data = deleted_data.sort_values(['UserId', 'UTCTimeOffset'])
    deleted_data.to_csv(join(source_pth,deleted_file), index=False)

print(f"Original data rows: {len(df)}")
print(f"Retained data rows: {len(final_df)}")
print(f"Deleted data rows: {len(deleted_data)}")
print(f"Number of trajectories with counts >= 10 deleted: {len(large_trajectories) - len(keep_large_df['trajectory_id'].unique())}")
########################### 3. Remove trajectories with count below a threshold, and remove user data with fewer than 10 trajectories ##############################

########################### 4. Divide trajectories based on user trajectory length, not considering whether trajectory_id is in the same count_id ###############################
import pandas as pd
import numpy as np

# Read the filtered result from the previous step
df = pd.read_csv(join(source_pth, '3_train_trajid_poi10.csv'))

# Check original data size
original_size = len(df)
print(f"Original data size (filtered_result.csv): {original_size}")

# Ensure data is sorted by UserId and UTCTimeOffset
df = df.sort_values(['UserId', 'UTCTimeOffset'])

# Initialize count_id column
df['count_id'] = -1  # Mark unassigned with -1

# Track global count_id
global_count_id = 0

# Process data for each user
for user_id in df['UserId'].unique():
    user_df = df[df['UserId'] == user_id].copy()
    total_count = len(user_df)

    if total_count <= 100:
        # For visit data <= 100, assign one count_id
        df.loc[df['UserId'] == user_id, 'count_id'] = global_count_id
        global_count_id += 1
    else:
        # Calculate number of groups, aiming for even distribution
        num_groups = max(2, int(np.ceil(total_count / 100)))  # At least 2 groups
        target_size = total_count // num_groups  # Target size per group, using integer division for evenness

        # Split data
        start_idx = 0
        for group in range(num_groups):
            # Calculate end point for current group
            if group == num_groups - 1:
                # Last group, assign all remaining data
                df.loc[user_df.index[start_idx:], 'count_id'] = global_count_id
                global_count_id += 1
                break
            else:
                # Calculate end position for current group
                end_idx = min(start_idx + target_size, len(user_df))
                # Adjust to trajectory_id boundary
                split_traj_id = user_df.iloc[end_idx - 1]['trajectory_id']
                traj_end_idx = user_df[user_df['trajectory_id'] == split_traj_id].index[-1]
                traj_end_pos = user_df.index.get_loc(traj_end_idx) + 1

                # Assign count_id
                df.loc[user_df.index[start_idx:start_idx + traj_end_pos], 'count_id'] = global_count_id
                global_count_id += 1
                start_idx += traj_end_pos

# Remap count_id to ensure continuity
unique_groups = sorted(df['count_id'].unique())
group_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_groups)}
df['count_id'] = df['count_id'].map(group_mapping)

# Verify data size remains unchanged
final_size = len(df[df['count_id'] != -1])
print(f"Processed data size: {final_size}")
if final_size != original_size:
    raise ValueError(f"Data size mismatch! Original: {original_size}, Processed: {final_size}")

# Save result
df.to_csv(join(source_pth, '4_final_result_with_count_id.csv'), index=False)

# Output statistics
print(f"Total users: {len(df['UserId'].unique())}")
print(f"Number of assigned count_ids: {df['count_id'].nunique()}")
# Check size distribution of each count_id
counts = df['count_id'].value_counts()
print("Distribution of records per count_id:")
print(counts)
########################### 4. Divide trajectories based on user trajectory length, not considering whether trajectory_id is in the same count_id ###############################

########################### 5. Construct POI pool for each count_id ################################################
# Read result from previous step
df = pd.read_csv(join(source_pth, '4_final_result_with_count_id.csv'))
# Check data size
print(f"Input data size: {len(df)}")

# Group by UserId to build POI pool for each user
poi_pools = []

for user_id in df['UserId'].unique():
    # Get all data for this user
    user_df = df[df['UserId'] == user_id]

    # Get number of groups for this user
    num_groups = user_df['count_id'].nunique()

    # Build POI pool: unique combination of PoiId and DetailedLocation
    # Use f-string to format as "PoiId (DetailedLocation)"
    poi_pool = user_df[['PoiId', 'DetailedLocation']].drop_duplicates()
    poi_pool_str = ", ".join(
        f"{row['PoiId']} ({row['DetailedLocation']})"
        for _, row in poi_pool.iterrows()
    )

    # Add POI pool information for each count_id of this user
    # Regardless of number of groups (1 or more), POI pool is the unique combination of all POIs visited by the user
    for count_id in user_df['count_id'].unique():
        poi_pools.append({
            'UserId': user_id,
            'count_id': count_id,
            'POI_Pool': poi_pool_str,  # Store as list
            'NumGroups': num_groups
        })

# Convert to DataFrame
poi_pool_df = pd.DataFrame(poi_pools)

# Save to new CSV file
poi_pool_df.to_csv(join(source_pth, '5_user_poi_pools.csv'), index=False)

# Output statistics
print(f"Total users: {len(df['UserId'].unique())}")
print(f"POI pool records: {len(poi_pool_df)}")
########################### 5. Construct POI pool for each count_id ################################################

## ###################### 6. Generate JSON description file ######################################################
# Input and output file names
input_file = "4_final_result_with_count_id.csv"  # Processed CSV file name
input_file_pool = "5_user_poi_pools.csv"  # Processed CSV file name
output_file = "6_json_rate0.5.json"  # Output text file name

# Read CSV files
data = pd.read_csv(join(source_pth, input_file))
data_pool = pd.read_csv(join(source_pth, input_file_pool))

# Ensure required columns exist
if not {'trajectory_id', 'UserId', 'UTCTimeOffset', 'PoiId', 'DetailedLocation', 'PoiCategoryId','count_id'}.issubset(data.columns):
    raise ValueError("CSV file missing required columns")

# Sort by UserId
data = data.sort_values(by=['UserId','UTCTimeOffset'])
# Convert UTCTimeOffsetEpoch to datetime format
data ['UTCTimeOffset'] = pd.to_datetime(data ['UTCTimeOffset'], unit='s')
# Format datetime to specified string format
data ['UTCTimeOffset'] = data['UTCTimeOffset'].dt.strftime('%Y-%m-%d %H:%M:%S')
# Group by trajectory_id
grouped_data = data.groupby('count_id')

# Store final result
result = []

# Process each trajectory_id
for count_id, group in grouped_data:
    # Generate trajectory description
    user_id = group['UserId'].iloc[0]  # User ID (assumes UserId is the same within each trajectory)
    trajectory_description = f"The following data is the check-in trajectory of user {user_id}:"

    # Concatenate visit records
    visits = []
    for idx, row in group.iterrows():
        visit_desc = f" At {row['UTCTimeOffset']}, user {user_id} visited POI id {row['PoiId']} which Detail Location and Category is {row['DetailedLocation']}."
        visits.append(visit_desc)

    # Concatenate trajectory description and all visit records
    trajectory_description += "".join(visits)
    # Generate POI pool description
    # Get POI_Pool for corresponding count_id from user_poi_pools.csv
    poi_pool_row = data_pool[(data_pool['UserId'] == user_id) &
                             (data_pool['count_id'] == count_id)]
    if poi_pool_row.empty:
        raise ValueError(f"No POI_Pool found for UserId {user_id} and count_id {count_id}")
    poi_pool = poi_pool_row['POI_Pool'].iloc[0]
    poi_pool_description = f"All POIs that the user has visited are [{poi_pool}]"

    count_id_total = len(group)
    requirements = f"""
Please organize your answer in a JSON object with the following structure:
{{
    "UserId": {user_id},
    "Countid": {count_id},
    "POI imputation":[
        {{
            "Insert_position": Between check-in at [TIMESTAMP1] (Poi id <PoiId>) and [TIMESTAMP2] (Poi id <PoiId>),
            "PoiID": <PoiId>
        }}
    ]
}}
After making each imputation suggestion, verify:
1. Does the sequence formed by the suggested POIs and the check-ins at both ends constitute a logically coherent and
contextually consistent activity flow? 
2. Is there sufficient time between check-ins for this additional visit? 
3. Does this suggestion align with the user's visitation patterns?
Only when all responses to the evaluation questions are affirmative can the suggested POI be deemed a valid completion.
If any check fails, reconsider your suggestion.
You must suggest POIs for user that are at least 50% the count total {count_id_total}.
"""
    # Add generated content to result
    result.append({
        "UserId": int(user_id),
        "count_id": count_id,
        "count_total": count_id_total,
        "All_trajectory_description": trajectory_description,
        "Visited_POI_pool": poi_pool_description,
        "requirements": requirements
    })

try:
    with open(join(source_pth, output_file), 'w', encoding="utf-8") as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)
except Exception as e:
    print(f"JSON serialization failed: {e}")
    # Fallback to simpler format or write item by item
    with open(join(source_pth, output_file) + '.safe', 'w', encoding="utf-8") as json_file:
        for item in result:
            try:
                json_file.write(json.dumps(item, ensure_ascii=False) + '\n')
            except Exception as e2:
                print(f"Item serialization failed: {e2}")

print(f"Processing complete, output file: {output_file}")
## ###################### 6. Generate JSON description file ######################################################

######################## 7. Deepseek visit ##############################################################################
import json
import jsonlines
import time
import random
from openai import OpenAI
from tqdm import tqdm

random.seed(42)

def process_json_file(input_file, output_file, api_key):
    start_time = time.time()
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    # client = OpenAI(
    #     api_key=api_key,base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # Read entire JSON array

    with jsonlines.open(output_file, mode="w") as writer:
        for entry in tqdm(data, desc="Processing prompts"):
            try:
                # Step 1: Build complete user input (key modification)
                request_content = "\n\n".join([
                    "# Historical Trajectory Description\n" + entry["All_trajectory_description"],
                    "# Available POI Pool\n" + entry["Visited_POI_pool"],
                    "# Task Requirements\n" + entry["requirements"]
                ])

                # Step 2: Construct API request body compliant with specifications
                response = client.chat.completions.create(
                    # model="deepseek-chat",
                    model="qwen-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content":"You are an expert in Point-of-Interest (POI) recommendation, you specialize in summarizing and analyzing users' periodic visitation patterns and behavioral paradigms. Since users' check-in behaviors are subjective, there exist instances where users forget to check in at certain locations they have visited, resulting in incomplete data. Your task is to leverage the existing \"All_trajectory_description\" of users and the \"Visited_POI_pool\" constructed from the POIs they have visited, to infer these potential POIs and fill in the gaps in their visit records follow the requirements in \"requirements\". After completing it, reflect on whether your results meet the \"requirements\". This imputation process requires sophisticated analysis of \n1. Consider the specific geographic locations and categories of the POIs that the user has visited in the All_trajectory_description to capture the user's historical preferences. People tend to visit similar locations at similar times or in similar areas. \n2.Consider physical barriers like rivers and highways that significantly extend travel time beyond straight-line distance when determining the likelihood of a user visiting certain POIs. \n3.Match POI categories to their typical visitation hours (e.g., nightclubs at night, breakfast spots in the morning) when imputing missing check-ins. \n4.Weekday versus weekend behavioral differences should be preserved when imputing missing visits, as users often follow distinct patterns based on day type. "
                        },
                        {
                            "role": "user",
                            "content": request_content  # Pass structured complete request
                        }
                    ],
                    # stream=True
                    response_format={"type": "json_object"},  # Force JSON format response
                    max_tokens = 8192,  # Set maximum output length to 8K
                    extra_body = {"chat_template_kwarges": {"enable_thinking":False}}  # Qwen requirement: disable thinking process
                )

                # Step 3: Parse and validate response results
                result = json.loads(response.choices[0].message.content)
                writer.write(result)

            except Exception as e:
                print(f"\nError processing entry {entry.get('count id')}: {str(e)}")
                # Optional: Write error entries to log
                with open("error_log.txt", "a") as log:
                    log.write(f"Error in entry {entry.get('count id', 'unknown')}: {str(e)}\n")

    print(f"\nProcessed {len(data)} entries. Time used: {time.time() - start_time:.2f}s")

file_pairs = [
    (join(source_pth, '6_json_rate0.5.json'),join(source_pth, '7_tky_qwen_json_output_rate0.5.json'))
]

api_key =  " " ##your api

# Batch processing
for input_path, output_path in file_pairs:
    process_json_file(input_path, output_path, api_key)
####################### 7. Deepseek visit ##############################################################################

######################### 8. Regenerate missing Deepseek outputs #####################################
import json
import random
import jsonlines
from openai import OpenAI
import time
from tqdm import tqdm

def load_json_file(filepath):
    """Load JSON file and return its contents."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl_file(filepath):
    """Load JSONL file and return its contents."""
    results = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            results.append(obj)
    return results

def find_missing_user_ids(trajectory_file, final_output_file):
    """Find UserIds in trajectory file that are not in final output file."""
    trajectory_data = load_json_file(trajectory_file)
    final_output_data = load_jsonl_file(final_output_file)

    # Extract UserIds from both files
    trajectory_user_ids = set(entry['UserId'] for entry in trajectory_data)
    final_output_user_ids = set(entry['UserId'] for entry in final_output_data)

    missing_user_ids = trajectory_user_ids - final_output_user_ids
    return missing_user_ids, trajectory_data

def generate_poi_recommendations(client, entry, recommendation_rate=0.4):
    """Generate POI recommendations for a given entry."""
    # Construct the request content similar to the original script
    request_content = "\n\n".join([
        "# Historical Trajectory Description\n" + entry["All_trajectory_description"],
        "# Available POI Pool\n" + entry["Visited_POI_pool"],
        "# Task Requirements\n" + entry["requirements"].replace("at least 50%",
                                                                f"at least {recommendation_rate * 100}%")
    ])

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            # model="qwen-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Point-of-Interest (POI) recommendation, specializing in analyzing users' visitation patterns. Your task is to infer potential POIs that the user might have visited but did not check in, following the provided requirements."
                },
                {
                    "role": "user",
                    "content": request_content
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=8192,
            # max_tokens=16384,
            extra_body = {"chat_template_kwarges": {"enable_thinking":False}}  # Qwen requirement: disable thinking process
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"Error generating recommendations for UserId {entry['UserId']}: {str(e)}")
        return None

def process_missing_user_ids(trajectory_file, final_output_file, output_file, api_key, recommendation_rate=0.5):
    """Process missing UserIds and generate recommendations."""
    start_time = time.time()

    # Set up OpenAI client (using DeepSeek)
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    # client = OpenAI(
    #     api_key=api_key,base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # Find missing UserIds and get trajectory data
    missing_user_ids, trajectory_data = find_missing_user_ids(trajectory_file, final_output_file)

    # Filter trajectory data to only include missing UserIds
    missing_trajectory_data = [entry for entry in trajectory_data if entry['UserId'] in missing_user_ids]

    # Open output file for writing
    with jsonlines.open(output_file, mode='w') as writer:
        for entry in tqdm(missing_trajectory_data, desc="Processing missing UserIds"):
            result = generate_poi_recommendations(client, entry, recommendation_rate)
            if result:
                writer.write(result)

    print(f"\nProcessed {len(missing_trajectory_data)} missing UserIds. Time used: {time.time() - start_time:.2f}s")

# Configuration
trajectory_file = join(source_pth, "6_json_rate0.5.json")
final_output_file =  join(source_pth, "7_json_output_rate0.5.json")
output_file =  join(source_pth, "8_missing_user_recommendations_0.5.json")

api_key = "" ##your api

# Run the script
process_missing_user_ids(trajectory_file, final_output_file, output_file, api_key)
######################### 8. Regenerate missing Deepseek outputs #####################################

############################## 9. Merge original and regenerated completions, with different count_ids for a user ########################################
import jsonlines
import os

def load_jsonl_file(filepath):
    """Load JSONL file and return its contents."""
    results = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            results.append(obj)
    return results

def merge_and_sort_recommendations(existing_file, new_recommendations_file, output_file):
    """
    Merge existing recommendations with new recommendations based on UserId and Countid,
    combining POI imputations, and sort by UserId and Countid.
    """
    # Load existing recommendations
    existing_recommendations = load_jsonl_file(existing_file)

    # Load new recommendations
    new_recommendations = load_jsonl_file(new_recommendations_file)

    # Create a dictionary to store all recommendations, keyed by (UserId, Countid)
    all_recommendations = {}

    # Add existing recommendations
    for rec in existing_recommendations:
        key = (rec['UserId'], rec['Countid'])
        all_recommendations[key] = rec

    # Add or update with new recommendations
    for new_rec in new_recommendations:
        key = (new_rec['UserId'], new_rec['Countid'])
        if key in all_recommendations:
            # If the (UserId, Countid) pair exists, append new POI imputations
            existing_poi = all_recommendations[key]['POI imputation']
            new_poi = new_rec['POI imputation']
            # Combine POI imputations, avoiding duplicates based on Insert_position and PoiID
            poi_set = {(poi['Insert_position'], poi['PoiID']) for poi in existing_poi}
            for poi in new_poi:
                if (poi['Insert_position'], poi['PoiID']) not in poi_set:
                    existing_poi.append(poi)
                    poi_set.add((poi['Insert_position'], poi['PoiID']))
        else:
            # If the (UserId, Countid) pair doesn't exist, add the new recommendation
            all_recommendations[key] = new_rec

    # Convert dictionary to list and sort by UserId and Countid
    sorted_recommendations = sorted(
        all_recommendations.values(),
        key=lambda x: (x['UserId'], x['Countid'])
    )

    # Write sorted recommendations to output file
    with jsonlines.open(output_file, mode='w') as writer:
        for rec in sorted_recommendations:
            writer.write(rec)

    print(f"Merged recommendations saved to {output_file}")
    print(f"Total recommendations: {len(sorted_recommendations)}")

# File paths
existing_file = join(source_pth,"7_json_output_rate0.5.json")
new_recommendations_file = join(source_pth,"8_missing_user_recommendations_0.5.json")
output_file = join(source_pth,"9_final_json_output_rate0.5.json")

# Run merge and sort
merge_and_sort_recommendations(existing_file, new_recommendations_file, output_file)
############################## 9. Merge original and regenerated completions, with different count_ids for a user ########################################

############### 10. Convert output to standard JSON format and standardize POIid ###################
import json
import jsonlines
import re
from datetime import datetime
import pytz

def convert_to_utc_timestamp(time_str):
    """Convert time string to UTC timestamp (seconds)"""
    if time_str == "end of trajectory":
        return -1
    try:
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        dt = pytz.utc.localize(dt)  # Assume original time is UTC
        return int(dt.timestamp())
    except ValueError:
        return -2 # Return -1 if conversion fails

def standardize_json(data):
    for user_data in data:
        if 'POI_imputation' in user_data:
            for poi in user_data['POI_imputation']:
                # Convert Insert_after and Insert_before to UTC timestamps
                if 'Insert_after' in poi:
                    poi['Insert_after'] = convert_to_utc_timestamp(poi['Insert_after'])
                if 'Insert_before' in poi:
                    poi['Insert_before'] = convert_to_utc_timestamp(poi['Insert_before'])
    return data

def convert_json_format(input_file, output_file):
    # Dictionary to hold the converted data
    result_dict = {}

    # Read the input JSONL file
    with jsonlines.open(input_file) as reader:
        for item in reader:
            user_id = item.get("UserId")
            count_id = item.get("Countid")
            poi_imputation_list = item.get("POI imputation", [])

            # Process each POI imputation
            processed_imputation = []
            for poi in poi_imputation_list:
                insert_position = poi.get("Insert_position", "")
                poi_id = poi.get("PoiID")

                # Extract time information
                insert_after = None
                insert_before = None

                if "Between check-in at" in insert_position:
                    parts = insert_position.split("Between check-in at")
                    if len(parts) > 1:
                        time_parts = parts[1].split("and")
                        if len(time_parts) > 1:
                            insert_after = time_parts[0].strip().split("(")[0].strip()
                            insert_before = time_parts[1].strip().split("(")[0].strip()

                # Only add if we successfully extracted both times
                if insert_after and insert_before:
                    processed_imputation.append({
                        "PoiID": poi_id,
                        "Insert_after": insert_after,
                        "Insert_before": insert_before
                    })

            # Add to result dictionary
            if user_id is not None:  # Ensure user_id exists
                result_dict[str(user_id)] = {
                    "user": user_id,
                    "count_id": count_id,
                    "POI_imputation": processed_imputation
                }

    # Convert result dictionary to list and sort by user ID
    result_list = [result_dict[key] for key in sorted(result_dict.keys(), key=lambda x: int(x))]

    processed_data = standardize_json(result_list)

    # Write to output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4)

    print(f"Conversion complete. Output written to {output_file}")
    print(f"Processed {len(result_list)} users")

# Example usage
input_file = join(source_pth,"9_final_json_output_rate0.5.json")  # Replace with your input file path
output_file = join(source_pth,"10_json_output_rate0.5_normal.json")  # Replace with your desired output file path
convert_json_format(input_file, output_file)
############### 10. Convert output to standard JSON format and standardize POIid ###################

############# 11. Round: Remove trajectories with count below a threshold #################################
input_file = join(source_pth,"3_train_trajid_poi10.csv")
# input_file = "./dataset/nyc_new_test_diff-poi/final_result_with_count_id.csv"
output_file = join(source_pth,"11_train_trajid_poi5_traj.csv")

# Read original data
df = pd.read_csv(input_file)

# 1. First, count the occurrences of each trajectory_id
trajectory_counts = df['trajectory_id'].value_counts()

# 2. Get trajectory_ids with counts >= 5, preserving original order
keep_large_df = df[df['trajectory_id'].isin(
    trajectory_counts[trajectory_counts >= 5].index)]
# print(trajectory_counts[trajectory_counts >= 10])

# Sort by UserId and UTCTimeOffset
final_df = keep_large_df.sort_values(['UserId', 'UTCTimeOffset'])
final_df.to_csv(output_file, index=False)
############# 11. Round: Remove trajectories with count below a threshold #################################

## ############# 12. Generate nearby_pois (train_nearby_pois_0.1km_pool20.csv) #################################
def read_and_clean_data(file_path):
    # Step 1: Read data from CSV
    df = pd.read_csv(file_path)
    print("Data Read Complete. Number of rows:", len(df))

    return df

def calculate_nearby_pois(df):
    # Step 3: Calculate Nearby POIs
    unique_pois = df[['PoiId', 'Latitude', 'Longitude','PoiCategoryName','DetailedLocation']].drop_duplicates()
    locations = unique_pois[['Latitude', 'Longitude']].to_numpy()

    nbrs = NearestNeighbors(radius=0.1 / 6371, algorithm='ball_tree').fit(np.radians(locations)) # radius=1 means within 1km

    def get_nearby_pois(lat, lon):
        lat_lon = np.radians([[lat, lon]])
        distances, indices = nbrs.radius_neighbors(lat_lon, return_distance=True)

        # Pair distances with POI indices and sort by distance
        nearby_with_distances = sorted(
            zip(distances[0], indices[0]),
            key=lambda x: x[0]
        )

        # Select up to 200 nearest POIs
        nearby = [
            f"{unique_pois.iloc[i]['PoiId']} ({unique_pois.iloc[i]['DetailedLocation']})"
            for _, i in nearby_with_distances[:20]
        ]
        return list(nearby)

    df['nearby_pois'] = df.apply(lambda row: get_nearby_pois(row['Latitude'], row['Longitude']), axis=1)
    df['nearby_pois'] = df['nearby_pois'].apply(lambda x: ', '.join(sorted(set(x))))
    df.to_csv(join(source_pth,'12_train_nearby_pois_0.1km_pool20.csv'), index=False)
    print("Nearby POIs Calculation Complete. Sample Data:")
    print(df[['Latitude', 'Longitude', 'nearby_pois']].head())
    return df

def main():
    file_path = join(source_pth,'11_train_trajid_poi5_traj.csv')  # CSV file with POI data
    df = read_and_clean_data(file_path)
    df = calculate_nearby_pois(df)
    return df
if __name__ == "__main__":
    main()
## ############# 12. Generate nearby_pois (train_nearby_pois_0.1km_pool20.csv) #################################

## ############# 13. Generate round JSON description file #################################
# Input and output file names
input_file = join(source_pth,"12_train_nearby_pois_0.1km_pool20.csv")  # Processed CSV file name
input_file_count = join(source_pth,"4_final_result_with_count_id.csv")  # Processed CSV file name
output_file = join(source_pth,"13_round_trajectory_0.1km_pool20_rate0.5.json")  # Output text file name

# Read CSV files
data = pd.read_csv(input_file)
data_count = pd.read_csv(input_file_count)

# Ensure required columns exist
if not {'trajectory_id', 'UserId', 'UTCTimeOffset', 'PoiId', 'DetailedLocation', 'PoiCategoryId',
        'nearby_pois'}.issubset(data.columns):
    raise ValueError("CSV file missing required columns")

if not {'count_id', 'UserId', 'UTCTimeOffset', 'PoiId', 'DetailedLocation', 'trajectory_id',
        'PoiCategoryId'}.issubset(data_count.columns):
    raise ValueError("CSV file missing required columns")
# Sort by UserId
data = data.sort_values(by=['UserId','UTCTimeOffset'])
# Convert UTCTimeOffsetEpoch to datetime format
data ['UTCTimeOffset'] = pd.to_datetime(data ['UTCTimeOffset'], unit='s')
# Format datetime to specified string format
data ['UTCTimeOffset'] = data['UTCTimeOffset'].dt.strftime('%Y-%m-%d %H:%M:%S')
# Group by trajectory_id
grouped_data = data.groupby('trajectory_id')
# print("grouped_data",grouped_data)

# Sort by UserId
data_count = data_count.sort_values(by=['UserId','UTCTimeOffset'])
# Convert UTCTimeOffsetEpoch to datetime format
data_count['UTCTimeOffset'] = pd.to_datetime(data_count['UTCTimeOffset'], unit='s')
# Format datetime to specified string format
data_count['UTCTimeOffset'] = data_count['UTCTimeOffset'].dt.strftime('%Y-%m-%d %H:%M:%S')
# count_data = data_count.groupby('count_id')

# Process final_result_with_count_id.csv
def build_count_id_trajectories(data_count):
    count_id_trajectories = {}
    for count_id, group in data_count.groupby('count_id'):
        user_id = group['UserId'].iloc[0]
        visits = []
        for idx, row in group.iterrows():
            visit_desc = f" At {row['UTCTimeOffset']}, user {user_id} visited POI id {row['PoiId']} which Detail Location and Category is {row['DetailedLocation']}."
            visits.append(visit_desc)
        trajectory_description = f"The following data is a trajectory of user {user_id}:" + "".join(visits)
        count_id_trajectories[count_id] = trajectory_description
    return count_id_trajectories

# Build trajectory descriptions for count_id
count_id_trajectories = build_count_id_trajectories(data_count)

# Store final result
result = []
# Process each trajectory_id
for trajectory_id, group in grouped_data:
    # Generate trajectory description
    user_id = group['UserId'].iloc[0]  # User ID (assumes UserId is the same within each trajectory)
    # trajectory_id = group['trajectory_id'].iloc[0]
    # Find corresponding count_id
    count_id = data_count[data_count['trajectory_id'] == trajectory_id]['count_id'].iloc[0]
    # Get corresponding trajectory description
    all_check_in = count_id_trajectories.get(count_id, "")

    # Generate description for current trajectory
    trajectory_description = f"The following data is a trajectory {trajectory_id} of user {user_id}:"
    # Concatenate visit records
    visits = []
    for idx, row in group.iterrows():
        visit_desc = f" At {row['UTCTimeOffset']}, user {user_id} visited POI id {row['PoiId']} which Detail Location and Category is {row['DetailedLocation']}."
        visits.append(visit_desc)

    # Concatenate trajectory description and all visit records
    trajectory_description += "".join(visits)

    # Get nearby POIs and remove duplicates
    nearby_pois = group['nearby_pois'].tolist()
    # Remove duplicate POIs (deduplicate by "PoiId" and "DetailedLocation")
    unique_nearby_pois = list(set(nearby_pois))
    if len(unique_nearby_pois)>40:
        unique_nearby_pois = unique_nearby_pois[:40]

    # Generate POI pool description
    poi_pool_description = "The POI pool within 0.1km around these visited POIs is [" + ", ".join(
        unique_nearby_pois) + "]"

    trajectory_total = len(group)
    requirements = f"""\
Please organize your answer in a JSON object with the following structure:
{{
    "UserId": {user_id},
    "trajectory_id": {trajectory_id},
    "POI imputation":[
        {{
            "Insert_position": Between check-in at [TIMESTAMP1] (Poi id <PoiId>) and [TIMESTAMP2] (Poi id <PoiId>),
            "PoiID": <PoiId>
        }}
    ]
}}
After making each imputation suggestion, verify:
1. Does the sequence formed by the suggested POIs and the check-ins at both ends constitute a logically coherent and
contextually consistent activity flow? 
2. Is there sufficient time between check-ins for this additional visit? 
3. Does this suggestion align with the user's visitation patterns?
Only when all responses to the evaluation questions are affirmative can the suggested POI be deemed a valid completion.
If any check fails, reconsider your suggestion.
You must suggest POIs for user that are at least 50% the count total {count_id_total}.
"""
    # Add generated content to result
    result.append({
        "UserId": int(user_id),
        "trajectory_id": trajectory_id,
        "trajectory_total":  trajectory_total,
        "trajectory_description": trajectory_description,
        "Nearby_POI_pool": poi_pool_description,
        "requirements": requirements
    })

# Output result as JSON file
try:
    with open(output_file, 'w', encoding="utf-8") as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)
except TypeError as e:
    print("JSON serialization error:", e)
    for item in result:
        print(type(item))  # Check type of each object

print(f"Processing complete, output file: {output_file}")
## ############# 13. Generate round JSON description file #################################

######################## 14. Deepseek round ##############################################################################
import json
import jsonlines
import time
import random
from openai import OpenAI
from tqdm import tqdm

def process_json_file(input_file, output_file, api_key):
    start_time = time.time()
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    # client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")  # SiliconFlow API

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # Read entire JSON array

    with jsonlines.open(output_file, mode="w") as writer:
        for entry in tqdm(data, desc="Processing prompts"):
            try:
                # Step 1: Build complete user input (key modification)
                request_content = "\n\n".join([
                    "# User\n" + str(entry["UserId"]),
                    "# trajectory_id" + str(entry["trajectory_id"]),
                    "# Specific Trajectory Description\n" + entry["trajectory_description"],
                    "# Available POI Pool\n" + entry["Nearby_POI_pool"],
                    "# Task Requirements\n" + entry["requirements"]
                ])
                # Step 2: Construct API request body compliant with specifications
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    # model="deepseek-reasoner",
                    messages=[
                        {
                            "role": "system",
                            "content":"You are a location data analysis expert specializing in identifying and completing missing check-ins within a single user trajectory. I will provide you with a description of a user trajectory and a nearby POI (Point of Interest) pool constructed from POIs within 0.1 km of the POIs historically visited by the user. Your task is to analyze the user's behavioral patterns within this trajectory and identify potential missing check-in POIs, offering reasonable completions. Users' check-in behavior is subjective,  so when analyzing the trajectory, pay special attention to the following scenarios that may lead to missing check-ins: 1.Short-Distance Consecutive Visit Scenario: After checking in at a previous POI, the user immediately proceeds to another nearby location but does not check-in. 2.Short-Stay Locations: The user visits certain locations briefly (e.g., convenience stores, coffee shops, fast-food restaurants), where check-ins are easily overlooked. 3.Routine Stops on Commute Routes: The user has regular stops along their daily commuting route but does not consistently check in at these locations. 4.Activity Continuity Disruption: The user's activities lack logical coherence, such as checking in directly from a residential location to a workplace and potentially missing intermediate locations like transportation hubs."
                        },
                        {
                            "role": "user",
                            "content": request_content  # Pass structured complete request
                        }
                    ],
                    # stream=True
                    response_format={"type": "json_object"},  # Force JSON format response
                    max_tokens = 8192  # Set maximum output length to 8K
                )
                result = json.loads(response.choices[0].message.content)
                writer.write(result)

            except Exception as e:
                print(f"\nError processing entry {entry.get('count id')}: {str(e)}")
                # Optional: Write error entries to log
                with open("error_log.txt", "a") as log:
                    log.write(f"Error in entry {entry.get('count id', 'unknown')}: {str(e)}\n")

    print(f"\nProcessed {len(data)} entries. Time used: {time.time() - start_time:.2f}s")

# Configuration parameters
file_pairs = [
    (join(source_pth,"13_round_trajectory_0.1km_pool20_rate0.5.json"),
     join(source_pth,"14_final_output_0.1km_pool20_rate0.51.json"))
]

api_key = ""   #your api
# Set random seed to ensure reproducible results
random.seed(42)  # Can choose any integer as seed value
# Batch processing
for input_path, output_path in file_pairs:
    process_json_file(input_path, output_path, api_key)
######################## 14. Deepseek round ##############################################################################

############### 15. Convert round output to standard JSON format and standardize POIid (with specific insertion time version) ###################
import json
import jsonlines
import re
from datetime import datetime
import pytz

def convert_to_utc_timestamp(time_str):
    """Convert time string to UTC timestamp (seconds)"""
    if time_str == "end of trajectory":
        return -1
    try:
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        dt = pytz.utc.localize(dt)  # Assume original time is UTC
        return int(dt.timestamp())
    except ValueError:
        return -2 # Return -1 if conversion fails

def standardize_json(data):
    for user_data in data:
        if 'POI_imputation' in user_data:
            for poi in user_data['POI_imputation']:
                # Convert Insert_after and Insert_before to UTC timestamps
                if 'Insert_after' in poi:
                    poi['Insert_after'] = convert_to_utc_timestamp(poi['Insert_after'])
                if 'Insert_before' in poi:
                    poi['Insert_before'] = convert_to_utc_timestamp(poi['Insert_before'])
    return data

def convert_json_format(input_file, output_file):
    # Dictionary to hold the converted data
    result_dict = {}

    # Read the input JSONL file
    with jsonlines.open(input_file) as reader:
        for item in reader:
            user_id = item.get("UserId")
            # count_id = item.get("Countid")
            traj_id = item.get("trajectory_id")
            poi_imputation_list = item.get("POI imputation", [])

            # Process each POI imputation
            processed_imputation = []
            for poi in poi_imputation_list:
                insert_position = poi.get("Insert_position", "")
                poi_id = poi.get("PoiID")

                # Extract time information
                insert_after = None
                insert_before = None

                if "Between check-in at" in insert_position:
                    parts = insert_position.split("Between check-in at")
                    if len(parts) > 1:
                        time_parts = parts[1].split("and")
                        if len(time_parts) > 1:
                            insert_after = time_parts[0].strip().split("(")[0].strip()
                            insert_before = time_parts[1].strip().split("(")[0].strip()

                # Only add if we successfully extracted both times
                if insert_after and insert_before:
                    processed_imputation.append({
                        "PoiID": poi_id,
                        "Insert_after": insert_after,
                        "Insert_before": insert_before
                    })
            if traj_id is not None:  # Ensure user_id exists
                result_dict[str(traj_id)] = {
                    "UserId": user_id,
                    "trajectory_id": traj_id,
                    "POI_imputation": processed_imputation
                }

    # Convert result dictionary to list and sort by user ID
    result_list = [result_dict[key] for key in sorted(result_dict.keys(), key=lambda x: int(x))]

    processed_data = standardize_json(result_list)

    # Write to output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4)

    print(f"Conversion complete. Output written to {output_file}")
    print(f"Processed {len(result_list)} users")

# Example usage
input_file = join(source_pth,"14_final_output_0.1km_pool20_rate0.5.json")  # Replace with your input file path
output_file = join(source_pth,"15_final_output_0.1km_pool20_rate0.5_normal.json")  # Replace with your desired output file path
convert_json_format(input_file, output_file)
############### 15. Convert round output to standard JSON format and standardize POIid (with specific insertion time version) ###################

############### 16. Complete dataset with final LLM inference results (round), converted to diff-poi version, based on time ###################
import pandas as pd
import random
import json

# Set random seed to ensure reproducible results
random.seed(42)

# Read original data CSV
original_data = pd.read_csv(join(source_pth,'4_final_result_with_count_id.csv'), encoding='utf-8')

# Read JSON data
with open(join(source_pth,'15_final_output_0.1km_pool20_rate0.5_normal.json'), 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# Prepare dictionary to store count_id corresponding to UserId and trajectory_id
user_traj_count = original_data.set_index(['UserId', 'trajectory_id'])['count_id'].to_dict()

# Prepare dictionary to store POI information
poi_info = {}
for _, row in original_data.iterrows():
    poi_info[row['PoiId']] = {
        'Latitude': row['Latitude'],
        'Longitude': row['Longitude'],
        'PoiCategoryName': row['PoiCategoryName'],
        'PoiCategoryId': row['PoiCategoryId'],
        'DetailedLocation': row['DetailedLocation']
    }

# Initialize new_data with all records from original data
new_data = original_data.to_dict(orient='records')

def calculate_time_offset(insert_after, insert_before):
    """Calculate insertion point time offset"""
    if insert_before == -1:
        # Add 10 minutes to after_time
        return int(insert_after) + 600  # 600 seconds = 10 minutes
    else:
        # Calculate average of after and before
        return (int(insert_after) + int(insert_before)) // 2

# Traverse JSON data to complete information
for entry in json_data:
    if 'UserId' not in entry or 'trajectory_id' not in entry or 'POI_imputation' not in entry:
        continue  # Skip empty or incorrectly formatted entries

    user_id = entry['UserId']
    trajectory_id = entry['trajectory_id']
    poi_imputations = entry['POI_imputation']

    # Select only 50% of POI data
    num_to_insert = int(len(poi_imputations) *0.2) # Take 50%, since it is already 50%, taking one-fifth results in r0.1
    poi_imputations = random.sample(poi_imputations, num_to_insert)  # Randomly select 50%

    # Get count_id related to trajectory_id and UserId
    count_id = user_traj_count.get((user_id, trajectory_id), None)
    if count_id is None:
        print(f"No count_id found for UserId {user_id} and trajectory_id {trajectory_id}")
        continue

    # Get original data related to trajectory_id
    trajectory_data = original_data[(original_data['UserId'] == user_id) &
                                  (original_data['trajectory_id'] == trajectory_id)].copy()

    # Convert to list for manipulation
    combined_data = trajectory_data.to_dict(orient='records')

    # Insert POIs in order
    for poi_entry in poi_imputations:
        poi_id = poi_entry['PoiID']
        insert_after = poi_entry['Insert_after']
        insert_before = poi_entry['Insert_before']

        if poi_id not in poi_info:
            print(f"No information found for PoiId {poi_id}, skipping")
            continue

        poi = poi_info[poi_id]

        new_time_offset = calculate_time_offset(insert_after, insert_before)

        # Create new row
        new_row = {
            'check_ins_id': -1,  # Placeholder
            'UTCTimeOffset': new_time_offset,  # Placeholder
            'UTCTimeOffsetEpoch': None,  # Placeholder
            'UserId': user_id,
            'Latitude': poi['Latitude'],
            'Longitude': poi['Longitude'],
            'PoiId': poi_id,
            'PoiCategoryId': poi['PoiCategoryId'],
            'PoiCategoryName': poi['PoiCategoryName'],
            'trajectory_id': trajectory_id,
            'DetailedLocation': poi['DetailedLocation'],
            'count_id': count_id
        }

        # Find insertion position
        if insert_before == -1:  # Insert at end of trajectory
            insert_idx = len(combined_data)
            for i, row in enumerate(combined_data):
                if row['UTCTimeOffset'] == insert_after:
                    insert_idx = i + 1
                    break
        else:
            insert_idx = 0
            for i, row in enumerate(combined_data):
                if row['UTCTimeOffset'] == insert_after:
                    insert_idx = i + 1
                    break
                elif row['UTCTimeOffset'] == insert_before:
                    insert_idx = i
                    break

        # Insert new row
        combined_data.insert(insert_idx, new_row)

    # Replace data for corresponding trajectory_id in new_data
    new_data = [row for row in new_data if not (row['UserId'] == user_id and row['trajectory_id'] == trajectory_id)] + combined_data

# Convert to DataFrame
new_dataframe = pd.DataFrame(new_data)
# Sort by UserId and trajectory_id
new_dataframe.sort_values(by=['UserId', 'trajectory_id','UTCTimeOffset'], inplace=True)
# Save to CSV file
new_dataframe.to_csv(join(source_pth,'16_final_round_complete_rate0.1_seed42.csv'), index=False, encoding='utf-8')

total_rows = len(new_dataframe)
print(f"Total rows: {total_rows}")
############### 16. Complete dataset with final LLM inference results (round), converted to diff-poi version, based on time ###################

############### 17. Complete dataset with final LLM inference results (visit), converted to diff-poi version, based on time ###################
import pandas as pd
import random
import json

# Set random seed to ensure reproducible results
random.seed(42)

# Read CSV data from first completion
original_data = pd.read_csv(join(source_pth,'4_final_result_with_count_id.csv'), encoding='utf-8')

# Read JSON data from second completion
with open(join(source_pth,'10_final_json_output_rate0.5_normal.json'), 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# Prepare dictionary to store information corresponding to UserId and count_id
user_count_id = original_data.set_index(['UserId', 'count_id']).index.to_list()  # Only need to know existing combinations

# Prepare dictionary to store POI information
poi_info = {}
for _, row in original_data.iterrows():
    poi_info[row['PoiId']] = {
        'Latitude': row['Latitude'],
        'Longitude': row['Longitude'],
        'PoiCategoryName': row['PoiCategoryName'],
        'PoiCategoryId': row['PoiCategoryId'],
        'DetailedLocation': row['DetailedLocation']
    }

# Initialize new_data with all records from original data
new_data = original_data.to_dict(orient='records')

def calculate_time_offset(combined_data, insert_after, insert_before, insert_idx):
    """Calculate insertion point time offset"""
    if insert_before == -1:
        # Insert at end, take last time plus 10 minutes
        last_time = combined_data[-1]['UTCTimeOffset'] if combined_data else insert_after
        return int(last_time) + 600  # 600 seconds = 10 minutes
    else:
        # Take average of time before insert_before and insert_before
        if insert_idx > 0:
            prev_time = combined_data[insert_idx - 1]['UTCTimeOffset']
            curr_time = insert_before
            return (int(prev_time) + int(curr_time)) // 2
        else:
            # If first position, take average of insert_after and insert_before
            return (int(insert_after) + int(insert_before)) // 2

# Traverse JSON data to complete information
for entry in json_data:
    if 'user' not in entry or 'count_id' not in entry or 'POI_imputation' not in entry:
        continue  # Skip empty or incorrectly formatted entries

    user_id = entry['user']
    count_id = entry['count_id']
    poi_imputations = entry['POI_imputation']

    # # Select only 50% of POI data
    # num_to_insert = len(poi_imputations)//2
    num_to_insert = int(len(poi_imputations)*0.6)
    poi_imputations = random.sample(poi_imputations, num_to_insert)  # Randomly select 50%
    if num_to_insert<1:
        num_to_insert =1
    filtered_poi_imputations = []
    for poi in poi_imputations:
        insert_after = poi['Insert_after']
        insert_before = poi['Insert_before']
        # Check time interval (if -1, treat as infinity, do not skip)
        if insert_before != -1 and (int(insert_before) - int(insert_after)) < 600:  # 600 seconds = 10 minutes
            continue  # Skip POIs with time interval less than 10 minutes
        filtered_poi_imputations.append(poi)
    poi_imputations = filtered_poi_imputations

    # Check if UserId and count_id exist
    if (user_id, count_id) not in user_count_id:
        print(f"No record found for UserId {user_id} and count_id {count_id}, skipping")
        continue

    # Get original data related to count_id
    trajectory_data = original_data[original_data['count_id'] == count_id].copy()

    # Convert to list and sort by time
    combined_data = trajectory_data.sort_values(by='UTCTimeOffset', na_position='last').to_dict(orient='records')

    # Insert POIs in order
    for poi_entry in poi_imputations:
        poi_id = poi_entry['PoiID']
        insert_after = poi_entry['Insert_after']
        insert_before = poi_entry['Insert_before']

        if poi_id not in poi_info:
            print(f"No information found for PoiId {poi_id}, skipping")
            continue

        poi = poi_info[poi_id]

        # Find insertion position (position before Insert_before) and get corresponding trajectory_id
        if insert_before == -1:
            insert_idx = len(combined_data)  # Insert at end
            traj_id = combined_data[-1]['trajectory_id'] if combined_data else 0  # Take last trajectory_id
        else:
            insert_idx = 0
            traj_id = 0  # Default value
            for i, row in enumerate(combined_data):
                if row['UTCTimeOffset'] == insert_before:
                    insert_idx = i
                    traj_id = row['trajectory_id']  # Use trajectory_id corresponding to Insert_before
                    break
            # If insert_before not found, try closest timestamp
            if insert_idx == 0:
                for i, row in enumerate(combined_data):
                    if row['UTCTimeOffset'] > insert_after:
                        insert_idx = i
                        traj_id = combined_data[i]['trajectory_id'] if i < len(combined_data) else combined_data[-1]['trajectory_id']
                        break
                else:
                    insert_idx = len(combined_data)
                    traj_id = combined_data[-1]['trajectory_id'] if combined_data else 0

        # Calculate new time offset
        new_time_offset = calculate_time_offset(combined_data, insert_after, insert_before, insert_idx)

        # Create new row
        new_row = {
            'check_ins_id': -2,  # Indicates second completion
            'UTCTimeOffset': new_time_offset,
            'UTCTimeOffsetEpoch': None,
            'UserId': user_id,
            'Latitude': poi['Latitude'],
            'Longitude': poi['Longitude'],
            'PoiId': poi_id,
            'PoiCategoryId': poi['PoiCategoryId'],
            'PoiCategoryName': poi['PoiCategoryName'],
            'trajectory_id': traj_id,  # Use trajectory_id determined by Insert_before
            'DetailedLocation': poi['DetailedLocation'],
            'count_id': count_id
        }

        # Insert new row
        combined_data.insert(insert_idx, new_row)

    # Replace data for corresponding count_id in new_data
    new_data = [row for row in new_data if row['count_id'] != count_id] + combined_data

# Convert to DataFrame
new_dataframe = pd.DataFrame(new_data)
# Sort by UserId, trajectory_id, and UTCTimeOffset, consistent with first completion
new_dataframe.sort_values(by=['UserId', 'trajectory_id', 'UTCTimeOffset'], na_position='last', inplace=True)
# Save to CSV file
new_dataframe.to_csv(join(source_pth,'17_train_all_v0.3_r0.1_seed42_time10m.csv'), index=False, encoding='utf-8')

total_rows = len(new_dataframe)
print(f"Total rows: {total_rows}")
############### 17. Complete dataset with final LLM inference results (visit), converted to diff-poi version, based on time ###################

############### 18. Insert trajectories with more than 10 POIs per day into the completed dataset ###################
import pandas as pd

######## Read two CSV files
train_file = join(source_pth,"17_train_all_v0.3_r0.1_seed42_time10m.csv")

deleted_file = join(source_pth,"3_1_deleted_traj_less_poi10.csv")

train_df = pd.read_csv(train_file)
deleted_df = pd.read_csv(deleted_file)

deleted_uids = set(deleted_df['UserId'])
train_uids = set(train_df['UserId'])

# Find UserIds in deleted_file that do not exist in train_file
missing_uids = deleted_uids - train_uids
if missing_uids:
    print(f"{len(missing_uids)} UserIds in deleted_file not found in train_file: {missing_uids}")

sb_df = train_df.dropna(subset=['UTCTimeOffset'])
print(len(sb_df))
print(len(train_df))
# Ensure trajectory_id column is numeric for sorting
train_df['trajectory_id'] = pd.to_numeric(train_df['trajectory_id'], errors='coerce')
deleted_df['trajectory_id'] = pd.to_numeric(deleted_df['trajectory_id'], errors='coerce')

# Create new DataFrame to store merged data
merged_df = pd.DataFrame(columns=train_df.columns)

# Traverse each UserId in train_df
for user_id, train_group in train_df.groupby('UserId'):
    # Get all data for this user
    train_data = train_group.to_dict('records')

    # Get data for same UserId in deleted_df
    deleted_data = deleted_df[deleted_df['UserId'] == user_id].to_dict('records')

    # Traverse each row in deleted_data
    for deleted_row in deleted_data:
        deleted_traj_id = deleted_row['trajectory_id']

        # Find first trajectory_id in train_data greater than deleted_traj_id
        insert_index = None
        for i, train_row in enumerate(train_data):
            train_traj_id = train_row['trajectory_id']

            # Skip if train_traj_id is null
            if pd.isna(train_traj_id):
                continue

            # If first trajectory_id greater than deleted_traj_id is found, record position and break
            if train_traj_id > deleted_traj_id:
                insert_index = i
                break

        # Insert data if position found
        if insert_index is not None:
            train_data.insert(insert_index, deleted_row)
        else:
            # If no insertion position found, append to end
            train_data.append(deleted_row)

    # Add this user's data to merged_df
    merged_df = pd.concat([merged_df, pd.DataFrame(train_data)], ignore_index=True)

# Append data for UserIds in deleted_file not present in train_file to merged_df
if missing_uids:
    missing_data = deleted_df[deleted_df['UserId'].isin(missing_uids)]
    merged_df = pd.concat([merged_df, missing_data], ignore_index=True)

# Drop DetailedLocation and count_id columns
merged_df = merged_df.drop(columns=['DetailedLocation','count_id'])

# Set UTCTimeOffset to 0 for check_ins_id of -1 or -2
merged_df.loc[merged_df['check_ins_id'].isin([-1, -2]), 'UTCTimeOffset'] = 0

merged_df['UTCTimeOffset'].fillna(0, inplace=True)
merged_df['UTCTimeOffset'] = merged_df['UTCTimeOffset'].astype(int)
merged_df.sort_values(by=['UserId','trajectory_id'], inplace=True)
# Save merged data to new file
merged_df.to_csv(join(source_pth,"18_train_final_v0.3_r0.1_seed42_time10m.csv"), index=False)
print("Merge complete, results saved to merged_output.csv")
total_rows = len(merged_df)
print(f"Total rows: {total_rows}")
############### 18. Insert trajectories with more than 10 POIs per day into the completed dataset ###################