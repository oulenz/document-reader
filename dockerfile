FROM python:3.5

ARG TFWRAPPER='git+ssh://git@github.com/epigramai/tfwrapper.git@pip-package'

WORKDIR /app

# Copying project files, remember to exclude large files.
# Each COPY line is a layer that will be cached by docker.
COPY /data ./data
COPY apify.py .

# Install libraries
COPY requirements.txt /app
RUN pip install -r requirements.txt

# Installing private dependencies
WORKDIR /root

COPY docker_resources .

# Install private dependencies without leaving secrets in image
RUN apt-get update

RUN wget -i pre_sign_url --quiet -O github_deployment_key_rsa && \
    chmod 600 github_deployment_key_rsa && \
    eval "$(ssh-agent -s)" && \
    ssh-add github_deployment_key_rsa && \
    mkdir .ssh && mv known_hosts .ssh/ && \
    pip --no-cache-dir install $TFWRAPPER && \
    rm github_deployment_key_rsa && \
    rm pre_sign_url && \
    rm -rf .ssh

# Make 5000 available to the world outside this container
EXPOSE 5000

# Python 3.5 onlbuild sets this as workingdir
WORKDIR /app

ENTRYPOINT ["gunicorn"]
CMD ["-b", "0.0.0.0:5000",  "apify:app"]