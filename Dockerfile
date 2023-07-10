FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . ${LAMBDA_TASK_ROOT}/.
RUN yum install -y libXext libSM libXrender mesa-libGL
RUN cd ${LAMBDA_TASK_ROOT}
RUN python3 setup.py

CMD [ "lambda_function.handler" ]