
The code is based on our pervious naming scheme of our model. Then, we changed a lot of terms that might cause confusion when referring to components in our model in order to give a better and clearer statement in our paper. Here is the list of terminology between the previously used and the currently used:

- `unconsciousness flow` (previous): `IGNN` (now)
- `consciousness flow`: `AGNN`
- `attended nodes`: nodes in the attending-from horizon
- `seen nodes`: nodes in the attending-to horizon
- `memorized nodes`: visited nodes
- `scanned edges`: edges of neighborhood

## Training and Evaluating

```bash
./run.sh 
```

<Dataset> can be one of 'Beauty','FB237', 'FB237_v2', 'FB15K', 'WN18RR', 'WN18RR_v2', 'WN', 'YAGO310', 'NELL995'.


