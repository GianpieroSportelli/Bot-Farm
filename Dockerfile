FROM python:3.6
LABEL maintainer="Gianpiero Sportelli <sportelligianpiero@gmil.com>"

COPY ./ /srv/src/
RUN pip3 install /srv/src
RUN rm -r /srv/src

ENTRYPOINT ["python3", "-m", "bot_farm.bot"]
CMD []