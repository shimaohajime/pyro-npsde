import modal 
import numpy as np 
from utils import gen_data
app = modal.App("yildiz-inference")
image = (
    modal.Image.debian_slim()
    .apt_install("libssl-dev")
    .pip_install_from_requirements("requirements.txt")
    .run_commands( # Install Python 3.5 manually since it's not available on Modal
        "apt-get update",
        "apt-get install -y git wget zlib1g-dev",
        "wget https://www.python.org/ftp/python/3.5.10/Python-3.5.10.tgz",
        "tar -xzf Python-3.5.10.tgz",
        "cd Python-3.5.10 && ./configure --enable-optimizations --with-ensurepip=upgrade && make altinstall",
        "ln -sf /usr/local/bin/python3.5 /usr/bin/python3",
        "ln -sf /usr/local/bin/pip3.5 /usr/bin/pip3",
        "cd .. && rm -rf Python-3.5.10 Python-3.5.10.tgz",
        "python3 -m pip install --upgrade pip",
    )

    .run_commands(
        "echo a",
        "git clone https://github.com/jinhongkuan/npde.git",
        "cd /npde && /usr/local/bin/python3.5 -m pip install -r requirements.txt --trusted-host pypi-mirror.modal.local",
    )
)

@app.function(image=image)
def train_model(t, Y, model='ode', sf0=1.0, ell0=[1.0, 1.0], W=6, ktype='id', whiten=False, num_iter=500, print_every=50, eta=0.02, plot=False):
    import subprocess
    import pickle
    import tempfile
    import os

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save t and Y as pickle files
        t_file = os.path.join(temp_dir, 't.pkl')
        Y_file = os.path.join(temp_dir, 'Y.pkl')
        
        with open(t_file, 'wb') as f:
            pickle.dump(t, f)
        with open(Y_file, 'wb') as f:
            pickle.dump(Y, f)

        # Prepare command
        command = [
            "python3.5", "/npde/runner.py", "train",
            "--t_file", t_file,
            "--Y_file", Y_file,
            "--model", model,
            "--sf0", str(sf0),
            "--ell0"] + [str(e) for e in ell0] + [
            "--W", str(W),
            "--ktype", ktype,
            "--num_iter", str(num_iter),
            "--print_every", str(print_every),
            "--eta", str(eta)
        ]

        if whiten:
            command.append("--whiten")
        if plot:
            command.append("--plot")

        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)

        # Check for errors
        if result.returncode != 0:
            raise Exception(f"Error running command: {result.stderr}")

        return result.stdout

@app.function(image=image)
def predict_model(model_file, x0, t):
    import subprocess
    import pickle
    import tempfile
    import os

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save x0 and t as pickle files
        x0_file = os.path.join(temp_dir, 'x0.pkl')
        t_file = os.path.join(temp_dir, 't.pkl')
        
        with open(x0_file, 'wb') as f:
            pickle.dump(x0, f)
        with open(t_file, 'wb') as f:
            pickle.dump(t, f)

        # Prepare output file
        output_file = os.path.join(temp_dir, 'prediction_output.pkl')

        # Prepare command
        command = [
            "python3.5", "/npde/runner.py", "predict",
            "--model_file", model_file,
            "--x0_file", x0_file,
            "--t_file", t_file,
            "--output_file", output_file
        ]

        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)

        # Check for errors
        if result.returncode != 0:
            raise Exception(f"Error running prediction: {result.stderr}")

        # Load and return prediction results
        with open(output_file, 'rb') as f:
            prediction_results = pickle.load(f)

        return prediction_results

@app.local_entrypoint()
def main():
    np.random.seed(918273) # just for illustration purposes
    x0,t,Y,X,D,f,g = gen_data('vdp', Ny=[35,40,30], tend=8, nstd=0.1)
    print(train_model.remote(t, Y))