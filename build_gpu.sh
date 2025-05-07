docker-compose -f docker-compose.gpu.yaml down
docker-compose -f docker-compose.gpu.yaml build
docker-compose -f docker-compose.gpu.yaml up -d
