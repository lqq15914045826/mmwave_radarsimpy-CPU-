# mmwave_radarsimpy(CPU)
## Establish vitural environment
```
conda create -n radar_cpu python=3.10
conda activate radar_cpu
```
## Generate mmwave data
```
python c_run_agent.py
```
The generated data will be stored in the \dataset folder.
## Visualization
```
python visualization.py
```
The generated images will be placed in the \vis_reports folder.

