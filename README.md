1. using the airflow 2.7 getting issues with 2.8 
2. initially user creation creates problem like user already exists then do user delete and create user it works 
3. I have mounted all volumes but getting error like permission 'C"// not able to create files because mlflow tries to create folders in the local that is not given permission to docker , for that 
        1. Find your container UID (usually Airflow runs as UID 50000):
        In local bash :     docker exec -it <airflow-container-name> id
        2.  On your host (local machine), run ( giving local folder permissions to docker container)
            sudo chown -R 50000:0 ./mlruns
            sudo chmod -R 775 ./mlruns

3.   Fix Directory Permissions (Run in your container)
        # Open a root shell in your container
        docker exec -it -u root salesconversions-airflow-1 bash

        # Now run these commands inside the container:
        chown -R airflow:root /opt/airflow/mlruns
        chmod -R 775 /opt/airflow/mlruns
        exit 
 # To run mlflow inside the container . 
4 .  mlflow ui --backend-store-uri mlruns --default-artifact-root mlruns --host 0.0.0.0 --port 5000
