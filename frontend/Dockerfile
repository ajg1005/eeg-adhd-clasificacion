FROM node:22-alpine

RUN mkdir -p /app && chown node:node /app
WORKDIR /app
USER node

COPY --chown=node:node frontend/package*.json ./
RUN npm ci

COPY --chown=node:node frontend .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
