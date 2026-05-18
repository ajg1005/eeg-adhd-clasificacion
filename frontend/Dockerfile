FROM node:22-alpine

WORKDIR /app

COPY frontend/package*.json ./

RUN npm ci --ignore-scripts

COPY frontend .

RUN mkdir -p /app/node_modules/.vite /tmp/vite \
    && chown -R node:node /app/node_modules/.vite /tmp/vite

USER node

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
