FROM python:3.6
LABEL maintainer="Gianpiero Sportelli <sportelligianpiero@gmil.com>"

COPY ./ /srv/src/
RUN pip3 install --no-cache-dir -r /srv/src/requirements-cpu.txt
RUN pip3 install /srv/src

ENTRYPOINT ["python3", "-m", "bot_farm.bot"]
CMD []