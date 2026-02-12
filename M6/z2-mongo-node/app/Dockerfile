FROM node:22-alpine AS builder

WORKDIR /usr/src/app

COPY package.json ./
RUN npm install

COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM node:22-alpine

RUN addgroup -S appgroup && adduser -S appuser -G appgroup

WORKDIR /usr/src/app

COPY package.json ./
RUN npm install --omit=dev

COPY --from=builder /usr/src/app/dist ./dist

USER appuser

EXPOSE 3000

CMD ["node", "dist/index.js"]
