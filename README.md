# LaMDA: LLM-Driven Multi-Perspective POI Completion for Next POI Recommendations
## 📦 Environment
1. Clone this repository to your local machine.

2. Install the enviroment by running
```bash
conda env create -f POI.yml
```

## 🔧 Data Processing and Training Steps

### Step 1: Data Formatting
We follow the data preprocessing method (including trajectory construction) from [**STHGCN**](https://github.com/alipay/Spatio-Temporal-Hypergraph-Model) and convert the dataset into the same format used by [**Diff-POI**](https://github.com/Yifang-Qin/Diff-POI).

### Step 2: POI Completion
Run the following script to process the raw dataset and generate POI-completed data:

```bash
python process_data_step.py 
```

### Step 3: Graph Construction
We build a dual-graph structure using the script:
```bash
python process_graph.py
```
⚠️ **Note:** The original and processed dataset is too large to be uploaded to the repository. You can download it from this link (https://pan.baidu.com/s/1UdFOvEhUPuCsmZ-pTzcLBw?pwd=d2j3)

### Step 4: Training LaMDA
To train the LaMDA model, run the following command:
```bash
python main.py --dataset tky --batch 1024 --patience 10 --dropout --lr 0.01 --gpu 0
```

## Additional Notes
We combine the Gowalla dataset provided in Diff-POI with the POI category information from STHGCN to determine the specific category of each POI. Using this information, we construct trajectory sequences and perform data preprocessing accordingly.
