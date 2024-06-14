Inference chain standalone
======================
This repo produces inference results, based on the provided tensorrt file and images (in rof format).
The final result is in rof format, along with the "prediction.rof" there is also an option to generate a video with the predictions.

## Setting up

We are using ADA inference image as our base image, hence a login to azure.registry might be required.

```sh
az login --tenant AADPACE.onmicrosoft.com --use-device-code
az acr login -n vdeepacrprod
```

Ater this, container can be started with with VScode Devcontainer's open folder in a container option.

### Setting up julia

Following lines of code is not a part of docker image since pf-2 cloud where this inference chain is going to be uploaded might already be equipped with julia. Hence for the moment these steps are to be perfomred manually, once inside the container.
```
wget --directory-prefix=/julia/ https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.0-linux-x86_64.tar.gz 

sudo tar zxvf /julia/julia-1.10.0-linux-x86_64.tar.gz -C /opt/ 

sudo ln -s /opt/julia-1.10.0/bin/julia /usr/bin/julia 

export JULIA_PKG_USE_CLI_GIT=true

```

After this type "julia" in the command prompt followed by "]" for the package mode. Install following packages for setting up julia

```
add StaticArrays

registry add https://oauth2:<git-token>@github.com/bosch-xcdx-data-platform/1PJuliaRegistry

add https://<nt-user>:<git-token>@github.com/bosch-xcdx-data-platform/GenericJuliaRofReader.git

add GenericRofReade

add Images, MLUtils, Tullio, Logging
```

press backspace followed by "exit()" for getting out of julia.

### Setting up python- packages

Following packages are outside of ADA-inference image and for the moment, need to be added manually.

```
pip install julia

pip install --index-url https://jfrog.ad-alliance.biz/artifactory/api/pypi/shared-pypi-dev/simple warper-package 

pip install pydantic-settings 
```
Finally, julia has to be installed inside python before we use the inference chain. In the terminal type python to open the command line python. Type follwing command after that.
```
 import julia
 julia.install()
```

## Using Inference chain

Start the "inference_chain.py" with neccessary paths to the rof file and trt model.