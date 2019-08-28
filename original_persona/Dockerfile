# Build this from the system repo
FROM ptf-system
ENV BUILDDEPS "git gcc libbz2-dev liblzma-dev"
RUN apt-get update && apt-get -y install --no-install-recommends $BUILDDEPS && rm -rf /var/lib/apt/lists*
ENV shell_dir /shell
WORKDIR "$shell_dir"
COPY persona "$shell_dir"
COPY requirements.txt "$shell_dir"

# Install any needed packages specified in requirements.txt
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
