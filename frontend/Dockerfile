FROM node:22-alpine AS build

WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci --ignore-scripts

COPY frontend .

# La URL del backend se resuelve en build time (Vite la inyecta en el bundle).
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build


FROM nginx:1.27-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
