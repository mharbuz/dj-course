# Stage 1: Build
FROM node:22-alpine AS build

WORKDIR /app

COPY package*.json .npmrc ./

# Install dependencies with cache mount
RUN --mount=type=cache,target=/root/.npm \
    npm ci

COPY . .
RUN npm run build

# Stage 2: Production
FROM nginx:alpine

COPY --from=build /app/dist/wms-frontend/ /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
