# Codebase Installation

This document outlines an installation guide for using the workbench.
See [DevelopersGuide.md](DevelopersGuide.md) for how to install the development dependencies.


### Step 0: Install Docker
If you are unsure which variant of docker install, we recommend [Docker Desktop](https://docs.docker.com/desktop/) to you.

* On windows, you might have to toggle docker to use _linux containers_
  1) right-click the Docker Desktop icon in your status bar
  2) click the _Switch to linux containers_ button
    * ![window_docker.png](docs%2Fimg%2Fwindow_docker.png)


### Step 1: Create a personal access token:
Currently, the images are deployed to a private GitLab registry.
Thus, we have to tell our docker installation how to download these.

* Create
  a [personal access token (PAT)](https://git.rwth-aachen.de/-/user_settings/personal_access_tokens)
  with `read_api` permissions (`read_registry` is not sufficient) on the
  git.rwth-aachen.de instance  (click the link for the correct location)
* It has to have access to the [wbgeo](https://git.rwth-aachen.de/wbgeo/) group and its subprojects
  * a project scoped token is not sufficient
* The same PAT can be used for the development guide.

### Step 2: Login with docker
Run `docker login registry.git.rwth-aachen.de` in a CLI and enter your GitLab username as the username, 
 and the personal access token as the password.

You should see a `Login succeeded` message.

### Step 3: Download the docker-compose.yml
Download the [docker-compose.yml](docker-compose.yml) file to your machine to a location of your choice.

This file tells docker how to orchestrate the containers required to run the workbench editor.

(In case you have already checked out the repository: 
You can use the docker-compose.yml directory in your git project instead)

### Step 4: Start the workbench editor
Switch to a CLI (e.g. cmd) and change into the directory of the downloaded file.
(If you are unsure where your file is:
Under windows: Shift-right-click the folder containing the file and select _Open in Terminal_.)

Run the `docker compose up` command in the CLI.

In case you receive an `error during connect` error: Ensure Docker (Desktop) is running.

### Step 5: Done
Open [http://localhost:8080](http://localhost:8080) in your browser.
You might have to wait for the console to pause its output/the backend to be ready.


## How to update the graphical interface:
Run `docker compose pull`, but step 4 should update the images automatically.  
Certain updates might require you to also update the _docker-compose.yml_ file,
 in which case you will have to perform steps 3 and 4 again.


## How to reset the graphical interface:
In case the application fails to start after an update, etc.:
Run `docker compose down` to reset the interface.
Then continue with step 4 to start the interface again



## How to use a local codebase
In your _docker-compose.yml_ file: Go to the `services.py_runner.volumes` key
and uncomment/add the following lines:

```yaml
services:
   py_runner:
      # ...
      volumes:
         - "./path/to/my/codebase/:/usr/src/app/codebase/"
```

Change the `./path/to/my/codebase/` path to point to codebase directory.

(Special case: If you are using the docker-compose.yml from the codebase's directory,
 uncomment option A instead.)

Now the container will start with your local codebase.
