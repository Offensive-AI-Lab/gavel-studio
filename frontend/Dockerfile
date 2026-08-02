# Node 20 LTS. Alpine for a lean image (~150 MB vs ~1 GB for the bookworm
# variant).
FROM node:20-alpine

WORKDIR /app

# Copy lockfile + manifest first so docker caches the (slow) npm ci layer.
# Subsequent rebuilds skip the install entirely until package*.json changes.
COPY package.json package-lock.json ./

# `npm ci` is the CI-correct install: errors if package-lock is out of
# sync with package.json, and never mutates the lockfile.
RUN npm ci

# Copy the rest of the frontend. In dev, the docker-compose volume mount
# shadows this so source changes show up live; the COPY keeps the image
# usable without compose.
COPY . .

EXPOSE 5173

# Default to Vite dev server, bound to all interfaces so the host port
# mapping reaches it. docker-compose.yml restates this as the command
# override for clarity, but having a CMD here means `docker run` works
# standalone too.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
