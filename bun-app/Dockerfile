FROM oven/bun:1.4.0-slim

WORKDIR /app

COPY package.json bun.lock ./
RUN bun install --frozen-lockfile --production

COPY src ./src

EXPOSE 3001

CMD ["bun", "src/index.js"]