FROM gp_docker_tensorflow:latest

add . .
workdir .
run pip install -r requirements.txt
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]