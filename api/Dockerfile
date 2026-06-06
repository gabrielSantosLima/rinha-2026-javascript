
FROM node:24-slim

WORKDIR /app

COPY package.json pnpm-lock.yaml ./

RUN npm install -g pnpm 
RUN pnpm install --no-frozen-lockfile --dangerously-allow-all-builds --config.confirmModulesPurge=false

COPY . .

EXPOSE 3333

CMD ["pnpm", "start"]
