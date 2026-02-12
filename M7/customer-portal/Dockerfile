# Stage 1: Build
FROM --platform=linux/amd64 node:22-slim AS build

WORKDIR /app

COPY package*.json ./

# Install native bindings + dependencies with cache mount
RUN --mount=type=cache,target=/root/.npm \
    npm install --save-dev @oxc-parser/binding-linux-x64-gnu lightningcss-linux-x64-gnu @rollup/rollup-linux-x64-gnu || true && \
    npm install

COPY . .

ENV NODE_ENV=production
RUN npm run build

# Stage 2: Production
FROM --platform=linux/amd64 node:22-alpine AS production

WORKDIR /app

ENV NODE_ENV=production
ENV NITRO_HOST=0.0.0.0
ENV NITRO_PORT=3000

# Nuxt .output is self-contained — no need for full node_modules
COPY --from=build /app/.output ./.output

EXPOSE 3000
USER node
CMD ["node", ".output/server/index.mjs"]
