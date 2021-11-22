# About

This folder contains a `docker-compose` file that we can use to run one or more docker images (think: applications) in an orechestrated way.

1.  A `Dockerfile` is a way that we can create a Docker image.  This provides granular control of a reproducible compute environment!

2.  A `docker-compose.yml` is a file that allows us to configure and run images.  

You can totally drive docker from the command line, but I try to stick with docker-compose files when possible, but it's not totally required.

`labelstudio`:  This is an awesome app to annotate our datasets of all sorts.  Many data projects in the real world require us to assemble our own datasets, and there may be times that we have to "label" our data with information that we can use in downstream modeling and reporting tasks.  


## Usage

This requires that you use the command line/terminal.  `cd` into the folder that has the `docker-compose.yml` file.

- `docker-commpose up` will run the app and log all of the output in the terminal.  This is particularly useful for debugging any issues with a config/setup.
- `docker-compose up -d` will run the app in a detached state, that is, the logs above are not shown.