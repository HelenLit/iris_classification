# Iris classification with PyTorch 
   
This project processes the [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set), splits it into training and inference sets, trains a classification model using Torch and performs batch inference. The entire process is containerized using Docker.  

## Notes  
   
Even though it is not recommended to store datasets in the git repository, I did so upon my mentor's recommendation for better code reproducibility. However, I have also included the script for data generation in `data_gen.py`.

## Project Structure  
   
```  
├── data/  
│   ├── iris_inference_data.csv  
│   ├── iris_test_data.csv  
│   ├── iris_train_data.csv  
├── data_process/  
│   ├── __init__.py  
│   ├── processing.py  
│   ├── scaler.pickle  
├── inference/  
│   ├── Dockerfile  
│   ├── run.py  
├── modeling/  
│   ├── nn_model.py  
├── models/  
├── results/   
├── training/  
│   ├── Dockerfile  
│   ├── train.py  
├── unittests/  
│   ├── after_inf_test.py  
│   ├── after_train_test.py  
│   ├── preinf_test.py  
│   ├── pretrain_test.py  
├── utils.py
├── settings.json
├── requirements.txt
├── data_gen.py  
```

## Settings:

The configurations for the project are managed using the `settings.json` file. It stores important variables that control the behaviour of the project. Examples could be the path to certain resource files, constant values, hyperparameters for an ML model, or specific settings for different environments.

Keep in mind that you may need to pass the path to your config to the scripts. For this, you may create a .env file or manually initialize an environment variable as `CONF_PATH=settings.json`.

Please note, some IDEs, including VSCode, may have problems detecting environment variables defined in the .env file. This is usually due to the extension handling the .env file. If you're having problems, try to run your scripts in a debug mode, or, as a workaround, you can hardcode necessary parameters directly into your scripts. Make sure not to expose sensitive data if your code is going to be shared or public. In such cases, consider using secret management tools provided by your environment.


## Installation  
   
### Prerequisites  

#### Docker Desktop
   
You can skip this step if you're already have Docker Desktop installed.

Installing Docker Desktop is a straightforward process. Head over to the Docker official website's download page ([Docker Download Page](https://www.docker.com/products/docker-desktop)), and select the version for your operating system - Docker Desktop is available for both Windows and Mac. After downloading the installer, run it, and follow the on-screen instructions. 

Once the installation is completed, you can open Docker Desktop to confirm it's running correctly. It will typically show up in your applications or programs list. After launching, Docker Desktop will be idle until you run Docker commands. This application effectively wraps the Docker command line and simplifies many operations for you, making it easier to manage containers, images, and networks directly from your desktop. 

Keep in mind that Docker requires you to have virtualization enabled in your system's BIOS settings. If you encounter issues, please verify your virtualization settings, or refer to Docker's installation troubleshooting guide. Now you're prepared to work with Dockerized applications!

#### MLFlow

MLFlow can be easily installed on a local machine using the pip, the Python package installer. To do so, open the command prompt (you can find it by searching for `cmd` in the Start menu) and type the following command:

```python
pip install mlflow
```

After the successful installation, you can start managing and deploying your ML models with MLFlow. For further information on how to use MLFlow at its best, refer to the official MLFlow documentation or use the `mlflow --help` command.

If you encounter any issues during the installation, you can bypass them by commenting out the corresponding lines in the `train.py` and `requirements.txt` files.

To run MLFlow, type `mlflow ui` in your terminal and press enter. If it doesn't work, you may also try `python -m mlflow ui`  This will start the MLFlow tracking UI, typically running on your localhost at port 5000. You can then access the tracking UI by opening your web browser and navigating to `http://localhost:5000`.
   
### Steps  
   
1. **Clone the repository:**  
    ```bash  
    git clone https://github.com/HelenLit/homework_module_8.git
    ```  
   
2. **Navigate to the project directory:**  
    ```bash  
    cd homework_module_8  
    ```  
   
3. **Build the Docker images for training:**
- Build the training Docker image. If the built is successfully done, it will automatically train the model:
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
- You may run the container with the following parameters to ensure that the trained model is here:
```bash
docker run -it training_image /bin/bash
```
- You can check model filename with running in docker container terminal:
```bash
ls models/
```
- Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using:
```bash
mkdir models
```
```bash
docker cp <container_id>:/app/models/<model_name>.pickle ./models
```
```bash
docker cp  <container_id>:/app/data_process/scaler.pickle ./data_process
```
Replace `<container_id>` with your running Docker container ID and `<model_name>.pickle` with your model's name.

- Alternatively, the `train.py` script can also be run locally as follows:

```bash
python3 training/train.py
```
Additionally, you can pass specify your own argument while running script above. Here's a summary of those arguments: 
- `--train_file`: Specify training data file (default is from `settings.json`).  
- `--test_file`: Specify test data file (default is from `settings.json`).  
- `--model_path`: Specify the path for the output model (default is `None`, which means it will use a timestamp).  
- `--epochs`: Number of training epochs (default is from `settings.json`).  
- `--learning_rate`: Learning rate for the optimizer (default is from `settings.json`).  
- `--dropout_rate`: Dropout rate for the model (default is from `settings.json`).  
- `--metric`: Evaluation metric (default is from `settings.json`).  
- `--max_rows_train`: Maximum number of rows for training the model (default is from `settings.json`).  
- `--max_rows_test`: Maximum number of rows for testing the model (default is from `settings.json`).  
- `--random_state`: Random state used for reproducibility (default is from `settings.json`).  
- `--scaler`: Name for scaler deserialization file (default is from `settings.json`).  
   
<hr>

4. **Build the Docker images for inference:**
- Build the inference Docker image:
```bash
docker build -f ./inference/Dockerfile --build-arg settings_name=settings.json -t inference_image .
```
- Run the inference Docker container:
```bash
docker run -it inference_image /bin/bash  
```
- Copy results to the `results` directory in your inference container:
```bash
docker cp <container_id>:/app/results/<result_file_name>.csv ./results
```
- Alternatively, you can also run the inference script locally:

```bash
python inference/run.py
```
## Unit tests

In this project, unit tests are implemented to verify the proper setup and functionality of various components such as directories, files, data processing, training, and inference. These tests ensure that each stage of the machine learning pipeline, from data preparation to model training and result generation, works correctly and reliably.
   
- pretrain_test.py  
This file contains unit tests that check the existence and contents of essential directories and files, and validate the data processing steps before the training process begins.  
   
- after_train_test.py  
This file contains unit tests that ensure the models directory exists and contains the trained model files after the training process is completed.  
   
- preinf_test.py  
This file contains unit tests that verify the presence and contents of necessary directories, files, and data processing components before the inference process starts.  
   
- after_inf_test.py  
This file contains unit test that confirm the results directory exists and is populated with output files after the inference process has been executed.

## Logs  
   
To find and view the logs of a running Docker container, you can follow these steps:  
   
1. **Open Docker Desktop:**  
   Ensure Docker Desktop is running on your machine. If it is not already running, launch Docker Desktop.  
   
2. **Navigate to the "Builds" Tab:**  
   In Docker Desktop, navigate to the "Builds" tab. This tab contains a list of all your Docker builds and running containers.  
   
3. **Select the Desired Build:**  
   From the list of builds, select the build corresponding to the container whose logs you wish to view. This will open the details of that specific build.  
   
4. **View Logs:**  
   Within the build details, there will be multiple tabs. Click on the "Logs" tab to view the logs of the selected container. Here, you can see all the logs generated by the container, including any output or error messages.
   

## Wrap Up
This project illustrates a simple, yet effective template to organize an ML project. Following good practices and principles, it ensures a smooth transition from model development to deployment.
