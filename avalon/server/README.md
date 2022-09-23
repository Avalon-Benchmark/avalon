
# Quick guide

To connect to the server, you can add the following to your `~/.ssh/config` and run `ssh avalon_controller`:
```
Host avalon_controller
    HostName node-010.int8.ai
    User user
    Port 49220
    IdentityFile ~/.ssh/science.ed25519
```

To sync any updates to the server:
```
scp -r -i ~/.ssh/science.ed25519 -P 49220 standalone/avalon/datagen/avalon_server/* "user@node-010.int8.ai:/opt/avalon_server"
```

To start the server run the following:
```
ROOT_PATH=/mnt/private/avalon export API_VERSION=<new_uuid_here> nohup python app.py > /mnt/private/logs/avalon/<new_uuid_here>.log 2>&1 &
```

To test if the server is running:

```
# useful to test if server is running
curl node-010.int8.ai:64080/info/
curl node-010.int8.ai:64080/get_state/
```

# How it works

TODO
* folder structure
* ignore list
* reset marker
* valid apk version
* when something is complete


# [OUTDATED] Commands
```
# start the server
ROOT_PATH=/mnt/private/avalon nohup python app.py > log.txt 2>&1 &

# useful to test if server is running
curl node-010.int8.ai:64080/info/
curl node-010.int8.ai:64080/get_state/

# sync changes
scp -r -i ~/.ssh/science.ed25519 -P 49220 standalone/avalon/datagen/avalon_server/* "user@node-010.int8.ai:/opt/avalon_server"
rsync -avzR --delete --filter=":- .gitignore" -e "ssh -i ~/.ssh/science.ed25519 -p 49220" . user@node-010.int8.ai:/opt/avalon_server

# ssh
ssh -i ~/.ssh/science.ed25519 -p 49220 user@node-010.int8.ai
```

```
ROOT_PATH=/tmp/avalon API_VERSION=debug python app.py

```
