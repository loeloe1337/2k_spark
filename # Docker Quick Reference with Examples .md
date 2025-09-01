# Docker Quick Reference with Examples & Scripts

---

## Containers

### Common Commands
- `docker ps`  
  List running containers.

- `docker ps -a`  
  List all containers (running + stopped).

- `docker stop <container_id_or_name>`  
  Stop a running container.

- `docker start <container_id_or_name>`  
  Start a stopped container.

- `docker restart <container_id_or_name>`  
  Restart a container.

- `docker rm <container_id_or_name>`  
  Remove a stopped container.

- `docker container prune --force --filter "until=48h"`  
  Remove stopped containers unused for more than 48 hours.

- `docker logs <container_id_or_name>`  
  View logs of a container.

- `docker exec -it <container_id_or_name> <command>`  
  Run a command inside a running container (e.g., bash shell).

- `docker inspect <container_id_or_name>`  
  Show detailed info about a container.

### Examples & Scripts

- Stop all running containers:
  ```bash
  docker stop $(docker ps -q)
  ```

- Remove all stopped containers unused for 2 days:
  
      This command removes all stopped containers that were created more than 48 hours ago.
      - `docker container prune --force --filter "until=48h"`
          - `--force`: Skips confirmation prompt.
          - `--filter "until=48h"`: Only prunes containers stopped more than 48 hours ago.
      Useful for cleaning up old containers and freeing disk space.
  
  - `docker container prune --force --filter "until=48h"`

- List running containers with numbers and stop container #2:
  ```bash
  docker ps --format "{{.ID}} {{.Names}}" | nl
  container_id=$(docker ps --format "{{.ID}} {{.Names}}" | sed -n '2p' | awk '{print $1}')
  docker stop $container_id
  ```

- Execute bash shell inside a running container:
  - `docker exec -it <container_id_or_name> bash`

## Images

### Common Commands
- `docker images`  
  List all images.

- `docker pull <image_name>`  
  Download an image from Docker Hub.

- `docker build -t <image_name> .`  
  Build an image from a Dockerfile in the current directory.

- `docker rmi <image_id_or_name>`  
  Remove an image.

- `docker image prune --force`  
  Remove dangling (unused) images.

- `docker image prune -a --force`  
  Remove all unused images (not referenced by any container).

### Examples & Scripts

- Pull an image from Docker Hub:
  - `docker pull nginx:latest`

- Build an image from Dockerfile in current directory:
  - `docker build -t myapp:latest .`

- Remove dangling images (untagged):
  - `docker image prune --force`

- Remove all unused images:
  - `docker image prune -a --force`

## Volumes

### Common Commands
- `docker volume ls`  
  List all volumes.

- `docker volume create <volume_name>`  
  Create a new volume.

- `docker volume inspect <volume_name>`  
  Show detailed info about a volume.

- `docker volume rm <volume_name>`  
  Remove a volume.

- `docker volume prune --force`  
  Remove all unused volumes.

### Examples & Scripts

- Create a volume:
  - `docker volume create mydata`

- Use a volume in a container:
  - `docker run -v mydata:/app/data myapp`

- Remove unused volumes:
  - `docker volume prune --force`

## Networks

### Common Commands
- `docker network ls`  
  List all networks.

- `docker network create <network_name>`  
  Create a new network.

- `docker network inspect <network_name>`  
  Show detailed info about a network.

- `docker network rm <network_name>`  
  Remove a network.

- `docker network prune --force`  
  Remove all unused networks.

### Examples & Scripts

- Create a custom network:
  - `docker network create mynet`

- Run a container attached to a custom network:
  - `docker run --network mynet myapp`

- Remove unused networks:
  - `docker network prune --force`