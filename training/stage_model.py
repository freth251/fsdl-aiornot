import argparse
from pathlib import Path
import tempfile

from models.resnet import ResnetModel
import torch
import wandb


api = wandb.Api()

DEFAULT_ENTITY = api.default_entity
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROD_STAGING_ROOT = PROJECT_ROOT / "models" / "artifacts"
LOG_DIR = Path("training") / "logs"
MODEL_CHECKPOINT_TYPE = "model"
BEST_CHECKPOINT_ALIAS = "best"
MODEL_CHECKPOINT_PATH = "model.ckpt"
STAGED_MODEL_TYPE = "prod-ready"  # we can choose the name of this type, and ideally it's different from checkpoints
STAGED_MODEL_FILENAME = (
    "model.pt"  # standard nomenclature; pytorch_model.bin is also used
)
DEFAULT_FROM_PROJECT = "aiornot-trace"
DEFAULT_TO_PROJECT = "aiornot-trace"
DEFAULT_STAGED_MODEL_NAME = "ResNet"


def main(args):
    prod_staging_directory = PROD_STAGING_ROOT / args.staged_model_name
    prod_staging_directory.mkdir(exist_ok=True, parents=True)
    entity = _get_entity_from(args)
    # if we're just fetching an already compiled model
    if args.fetch:
        # find it and download it
        print("fetching artifact")
        staged_model = f"{entity}/{args.from_project}/{args.staged_model_name}:latest"
        artifact = download_artifact(staged_model, prod_staging_directory)
        print_info(artifact)
        return  # and we're done

    # otherwise, we'll need to download the weights, compile the model, and save it
    with wandb.init(
        job_type="stage", project=args.to_project, dir=LOG_DIR
    ):  # log staging to W&B so prod and training are connected
        # find the model checkpoint and retrieve its artifact name and an api handle
        ckpt_at, ckpt_api = find_artifact(
            entity,
            args.from_project,
            type=MODEL_CHECKPOINT_TYPE,
            alias=args.ckpt_alias,
            run=args.run,
        )

        # get the run that produced that checkpoint
        logging_run = get_logging_run(ckpt_api)
        print_info(ckpt_api, logging_run)
        metadata = get_checkpoint_metadata(logging_run, ckpt_api)

        # create an artifact for the staged, deployable model
        staged_at = wandb.Artifact(
            args.staged_model_name, type=STAGED_MODEL_TYPE, metadata=metadata
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            # download the checkpoint to a temporary directory
            download_artifact(ckpt_at, tmp_dir)
            # reload the model from that checkpoint
            model = load_model_from_checkpoint(metadata, directory=tmp_dir)
            # save the model to torchscript in the staging directory
            save_model_to_torchscript(model, directory=prod_staging_directory)

        # upload the staged model so it can be downloaded elsewhere
        upload_staged_model(staged_at, from_directory=prod_staging_directory)


def _find_artifact_run(entity, project, type, run, alias):
    run_name = f"{entity}/{project}/runs/{run}"
    api_run = api.run(run_name)
    artifacts = api_run.logged_artifacts()
    match = [art for art in artifacts if alias in art.aliases and art.type == type]
    if not match:
        raise ValueError(
            f"No artifact with alias {alias} found at {run_name} of type {type}"
        )
    if len(match) > 1:
        raise ValueError(
            f"Multiple artifacts ({len(match)}) with alias {alias} found at {run_name} of type {type}"
        )
    return f"{entity}/{project}/{match[0].name}"


def _find_artifact_project(entity, project, type, alias):
    project_name = f"{entity}/{project}"
    api_project = api.project(project, entity=entity)
    api_artifact_types = api_project.artifacts_types()
    # loop through all artifact types in this project
    for artifact_type in api_artifact_types:
        print(artifact_type)
        if artifact_type.name != type:
            continue  # skipping those that don't match type
        collections = artifact_type.collections()
        # loop through all artifacts and their versions
        for collection in collections:
            versions = collection.versions()
            for version in versions:
                if (
                    alias in version.aliases
                ):  # looking for the first one that matches the alias
                    return f"{project_name}/{version.name}"
        raise ValueError(
            f"Artifact with alias {alias} not found in type {type} in {project_name}"
        )
    raise ValueError(
        f"Artifact type {type} not found. {project_name} could be private or not exist."
    )


def find_artifact(entity: str, project: str, type: str, alias: str, run=None):
    print(f"find artifact with entity:{entity}, type:{type}, alias:{alias}")
    if run is not None:
        path = _find_artifact_run(entity, project, type=type, run=run, alias=alias)
    else:
        path = _find_artifact_project(entity, project, type=type, alias=alias)
    return path, api.artifact(path)


def get_logging_run(artifact):
    api_run = artifact.logged_by()
    return api_run


def get_checkpoint_metadata(run, checkpoint):
    config = run.config
    out = {"config": config}
    try:
        ckpt_filename = checkpoint.metadata["original_filename"]
        out["original_filename"] = ckpt_filename
        metric_key = checkpoint.metadata["ModelCheckpoint"]["monitor"]
        metric_score = checkpoint.metadata["score"]
        out[metric_key] = metric_score
    except KeyError:
        pass
    return out


def download_artifact(artifact_path, target_directory):
    """Downloads the artifact at artifact_path to the target directory."""
    if (
        wandb.run is not None
    ):  # if we are inside a W&B run, track that we used this artifact
        artifact = wandb.use_artifact(artifact_path)
    else:  # otherwise, just download the artifact via the API
        artifact = api.artifact(artifact_path)
    artifact.download(root=target_directory)

    return artifact


def load_model_from_checkpoint(ckpt_metadata, directory):
    print(f"load_model_from_checkpoint at dir={directory}")
    config = ckpt_metadata["config"]
    args = argparse.Namespace(**config)

    model = ResnetModel()

    # load LightningModule from checkpoint
    pth = Path(directory) / "model.ckpt"
    lit_model = ResnetModel.load_from_checkpoint(
        checkpoint_path=pth, args=args, model=model, strict=False
    )
    lit_model.eval()

    return lit_model


def upload_staged_model(staged_at, from_directory):
    print(f"upload_staged_model from dir={from_directory}")
    staged_at.add_file(Path(from_directory) / "model.pt")
    wandb.log_artifact(staged_at)


def save_model_to_torchscript(model, directory):
    print(f"save_model_to_torchscript to dir={directory}")
    scripted_model = model.to_torchscript(method="script", file_path=None)
    path = Path(directory) / STAGED_MODEL_FILENAME
    torch.jit.save(scripted_model, path)


def _get_entity_from(args):

    entity = args.entity
    if entity is None:
        raise RuntimeError(
            f"No entity argument provided. Use --entity=DEFAULT to use {DEFAULT_ENTITY}."
        )
    elif entity == "DEFAULT":
        entity = DEFAULT_ENTITY

    return entity


def print_info(artifact, run=None):
    if run is None:
        run = get_logging_run(artifact)

    full_artifact_name = f"{artifact.entity}/{artifact.project}/{artifact.name}"
    print(f"Using artifact {full_artifact_name}")
    artifact_url_prefix = f"https://wandb.ai/{artifact.entity}/{artifact.project}/artifacts/{artifact.type}"
    artifact_url_suffix = f"{artifact.name.replace(':', '/')}"
    print(f"View at URL: {artifact_url_prefix}/{artifact_url_suffix}")

    print(f"Logged by {run.name} -- {run.project}/{run.entity}/{run.id}")
    print(f"View at URL: {run.url}")


def _setup_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help=f"If provided, check ENTITY/FROM_PROJECT for an artifact with the provided STAGED_MODEL_NAME and download its latest version to {PROD_STAGING_ROOT}/STAGED_MODEL_NAME.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help=f"Entity from which to download the checkpoint. Note that checkpoints are always uploaded to the logged-in wandb entity. Pass the value 'DEFAULT' to also download from default entity, which is currently {DEFAULT_ENTITY}.",
    )
    parser.add_argument(
        "--from_project",
        type=str,
        default=DEFAULT_FROM_PROJECT,
        help=f"Project from which to download the checkpoint. Default is {DEFAULT_FROM_PROJECT}",
    )
    parser.add_argument(
        "--to_project",
        type=str,
        default=DEFAULT_TO_PROJECT,
        help=f"Project to which to upload the compiled model. Default is {DEFAULT_TO_PROJECT}.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help=f"Optionally, the name of a run to check for an artifact of type {MODEL_CHECKPOINT_TYPE} that has the provided CKPT_ALIAS. Default is None.",
    )
    parser.add_argument(
        "--ckpt_alias",
        type=str,
        default=BEST_CHECKPOINT_ALIAS,
        help=f"Alias that identifies which model checkpoint should be staged.The artifact's alias can be set manually or programmatically elsewhere. Default is {BEST_CHECKPOINT_ALIAS!r}.",
    )
    parser.add_argument(
        "--staged_model_name",
        type=str,
        default=DEFAULT_STAGED_MODEL_NAME,
        help=f"Name to give the staged model artifact. Default is {DEFAULT_STAGED_MODEL_NAME!r}.",
    )
    return parser


if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()
    main(args)
