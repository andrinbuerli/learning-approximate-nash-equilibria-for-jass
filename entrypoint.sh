#!/bin/bash

BASEDIR=${PWD}

export WANDBKEY=$(cat .wandbkey)

echo "Found base dir at ${BASEDIR}"

echo "Checking jass-kit-py"
cd /repos/jass-kit-py
git pull

echo "Checking jass-ml-py"
cd /repos/jass-ml-py
git pull

echo "Checking jass-kit-cpp"
cd /repos/jass-kit-cpp
git fetch
UPSTREAM='@{u}'
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse "$UPSTREAM")
BASE=$(git merge-base @ "$UPSTREAM")

if [ $LOCAL = $REMOTE ]; then
    echo "Up-to-date"
elif [ $LOCAL = $BASE ]; then
    echo "Need to pull"
    pip uninstall jasscpp -y
    git pull
    pip install .
elif [ $REMOTE = $BASE ]; then
    echo "Need to push"
else
    echo "!!! ERROR: Branch Diverged !!!"
fi

echo "Changing to base dir at ${BASEDIR}"
cd ${BASEDIR}

exec "$@"