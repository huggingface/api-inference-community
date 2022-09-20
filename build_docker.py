#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
import uuid


def run(command):
    print(" ".join(command))
    p = subprocess.run(command)
    if p.returncode != 0:
        sys.exit(p.returncode)


def build(framework: str):
    DEFAULT_HOSTNAME = os.getenv("DEFAULT_HOSTNAME")
    hostname = DEFAULT_HOSTNAME
    tag_id = str(uuid.uuid4())[:5]
    tag = f"{framework}-{tag_id}"
    container_tag = f"{hostname}/api-inference-prod-community:{tag}"

    command = ["docker", "build", f"docker_images/{framework}", "-t", container_tag]
    run(command)

    command = ["aws", "ecr", "get-login-password"]
    ecr_login = subprocess.Popen(command, stdout=subprocess.PIPE)
    docker_login = subprocess.Popen(
        ["docker", "login", "-u", "AWS", "--password-stdin", hostname],
        stdin=ecr_login.stdout,
        stdout=subprocess.PIPE,
    )
    docker_login.communicate()

    command = ["docker", "push", container_tag]
    run(command)
    return tag


def main():
    frameworks = {
        dirname for dirname in os.listdir("docker_images") if dirname != "common"
    }
    framework_choices = frameworks.copy()
    framework_choices.add("all")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "framework",
        type=str,
        choices=framework_choices,
        help="Which framework image to build.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Should we try and update the production",
    )
    args = parser.parse_args()

    branch = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .strip()
    )
    if branch != "main":
        raise Exception(f"Go to branch `main` ({branch})")

    print("Pulling")
    subprocess.run(["git", "pull"])

    if args.framework == "all":
        outputs = []
        for framework in frameworks:
            tag = build(framework)
            outputs.append((framework, tag))

        for (framework, tag) in outputs:
            print(f"{framework.upper()}_CPU_TAG", tag)

    else:
        tag = build(args.framework)
        outputs = [(args.framework, tag)]

    if args.deploy:
        values = [
            f"models_tag.{framework.upper()}_CPU_TAG={tag}"
            for framework, tag in outputs
        ]
        cmd = [
            "helm",
            "upgrade",
            "prod",
            "chart/",
            "--reuse-values",
            "--set",
            ",".join(values),
        ]
        run(cmd)


if __name__ == "__main__":
    main()
