## Activate Virtual Environment

```
course_checks_env\Scripts\activate.bat
```

## Install Requirement.txt

```
pip install -r src\requirements.txt
```

## Deactivate Virtual Environment

```
deactivate
```

## Sequence of Execution

```
pull_datasets.py --> preprocessing.py --> predict.py --> Powerbi/Oracle Dashboard --> Send Emails
```