FROM julia:1.10.2

ENV USER pluto
ENV JULIA_NUM_THREADS 100


ENV USER_HOME_DIR /home/${USER}
ENV JULIA_DEPOT_PATH ${USER_HOME_DIR}/src/.julia


RUN useradd -m -d ${USER_HOME_DIR} ${USER} && chown -R ${USER} ${USER_HOME_DIR}

COPY . ${USER_HOME_DIR}/src
RUN chown -R ${USER} ${USER_HOME_DIR}/src

WORKDIR ${USER_HOME_DIR}/src

RUN mkdir -p /home/pluto/src/.julia/environments/v1.10/ &&\
    cp Manifest.toml /home/pluto/src/.julia/environments/v1.10/Manifest.toml &&\
    cp Project.toml /home/pluto/src/.julia/environments/v1.10/Project.toml &&\
    julia --project=. -e "import Pkg; Pkg.activate(); Pkg.instantiate(); Pkg.precompile();" &&\
    chown -R ${USER} ${USER_HOME_DIR} &&\
    chmod -R g+w ${USER_HOME_DIR}

RUN apt-get update && apt-get install -y vim

USER ${USER}

EXPOSE 1234

CMD [ "julia", "--project=/home/pluto/src/", "-e", "import Pluto; Pluto.run(host=\"0.0.0.0\", port=1234, launch_browser=false, require_secret_for_open_links=false, require_secret_for_access=false)"]

