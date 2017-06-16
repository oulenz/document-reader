#!/usr/bin/env bash

REPO="epigramai/document_scanner"

aws s3 presign --profile stian-docker s3://epigram-harry-potter/docker-key --expires-in 300 > docker_resources/pre_sign_url

if [ -z "$1" ]
    then
        echo "No tag specific tag provided. Building to $REPO:latest" && \
        docker build --tag $REPO .
else
    echo "Tag '$1' provided, builing to $REPO:$1"
    docker build --tag $REPO":$1" .
fi
