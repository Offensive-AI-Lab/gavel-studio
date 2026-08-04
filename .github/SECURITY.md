# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue.

Use GitHub's private reporting: open the **Security** tab of this repository and
choose **Report a vulnerability**. Only the maintainers can see the report,
which gives us time to fix the problem before it is public.

Helpful things to include:

* what an attacker can do, and what they need to get there,
* the steps to reproduce it,
* the version or commit you tested,
* how you were running Studio (Docker, native, remote GPU worker).

We will confirm we received your report and keep you updated while we work on
it. Please give us a chance to release a fix before writing about it publicly.

## What is covered

The code in this repository: the backend, the interface, and the GPU worker.

The public rule library is a separate project. Report problems with its content
in [gavel-rules](https://github.com/Offensive-AI-Lab/gavel-rules).

## Versions

This is a research tool under active development. Fixes go to the latest
`main`; there are no maintained older releases.

## What we do want to hear about

Studio is a single-user application that runs on your own machine, with no
login and no access control by design. Reports that assume otherwise — or that
depend on already having access to the machine — are out of scope. Things that
are in scope, for example:

* content synced from the public library that can run code, write outside its
  directory, or reach the local network when Studio processes it,
* a request to the backend that reads or writes files outside its workspace,
* a way for one Studio installation to leak its credentials to a remote host,
* a flaw in how the backend authenticates to a remote GPU worker.
