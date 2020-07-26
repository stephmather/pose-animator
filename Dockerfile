#Build using
# docker build -t pose_animator .
#run using 
# docker run -i -t -p 1234:1234 pose_animator


FROM node 
WORKDIR /app 
COPY . /app
RUN rm /app/package-lock.json
RUN rm /app/yarn.lock
RUN yarn
CMD yarn docker
EXPOSE 1234
