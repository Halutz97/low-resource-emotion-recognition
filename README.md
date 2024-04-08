# low-resource-emotion-recognition

# HPC SETUP NOTES

First run the following command on your command line using your student id\
`ssh <student_id>@login.hpc.dtu.dk`

Then load modules by running the following commands\
`module load python3/3.10.12`\
`module load cuda/12.1`

Then check if the modules loaded correctly by running the following command (you should see python and cuda listed)\
`module list`

Then create a directory in your environment \
run `ls` to see available directories and then run `cd Desktop` to go into desktop\
then run `mkdir emotion_recognition` to create a directory for our project. Run `cd emotion_recognition` to go into this folder

Then we will create a virtual environment in this directory for the project. For this run the following command\
`python3 -m venv .venv` (make sure you are in the directory of car-seg as it will be easier to have the venv in the same directory as the project)

Now activate the virtual environment by running `source .venv/bin/activate` (still inside the car-seg directory). You should see (.venv) in front of your command line prompt, this means the virtual environment is active. \
You can deactivate by running `deactivate` and activate back using `source .venv/bin/activate` again.

#### Installing packages - example

Then run the following to install pytorch while venv is active \
`pip3 install torch torchvision torchaudio`\
`python -m pip install -U matplotlib`\
Any other package can also be installed the same way while the venv is active

#### Transferring data

Now that we have the environment we will transfer our data onto the server as well\
For this open a new terminal on your own local environment and run the following command\
You need to replace the first one with the path in your own computer and also change the student id in the second part\
`scp -r /Users/arime/Desktop/thesis/data s163956@transfer.gbar.dtu.dk:~/Desktop/emotion_recognition`\
Now you should wait until everything is copied on

Now that we have the data we can clone the git repo on there\
I used the https clone method with a personal token\
run `git clone https://github.com/Halutz97/low-resource-emotion-recognition.git`\

#### Note that it is maybe easier to just copy files one by one manually!

Now we should have the venv, the data and the code setup to run training


#### WORKING AFTER INITIAL SETUP
First run the following command on your command line using your student id \
`ssh <student_id>@login.hpc.dtu.dk `

Then load modules by running the following commands\
`module load python3/3.10.12`\
`module load cuda/12.1`

Go into the project dir\
`cd Desktop/emotion_recognition`

Run the following to activate venv \
`source .venv/bin/activate`

Start working
