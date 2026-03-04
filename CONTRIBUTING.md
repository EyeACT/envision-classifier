# Contributor Guide

## Setup

### Requirements

* Make:
  - macOS: `$ xcode-select --install`
  - Linux: [https://www.gnu.org](https://www.gnu.org/software/make)
  - Windows: `$ choco install make` [https://chocolatey.org](https://chocolatey.org/install)
* Python: `$ asdf install` (https://asdf-vm.com)[https://asdf-vm.com/guide/getting-started.html]
* Poetry: [https://python-poetry.org](https://python-poetry.org/docs/#installation)

To confirm these system dependencies are configured correctly:

```text
$ make doctor
```

### Installation

Install project dependencies into a virtual environment:

```text
$ make install
```

## Development Tasks

Run the tests:

```text
$ make test
```

Run static analysis:

```text
$ make check
```

Build the documentation:

```text
$ make docs
```

## Release Tasks

Release to PyPI:

```text
$ make upload
```
