DESCRIPTION
--------------------
Active Anomaly Detection (AAD) with process mining enables end-users to both gain insight in the process of the client and adjust the weights of individual models in the ensemble method at the same time. In main.py, the AAD framework is instantiated with an Isolation Forest.


INSTALL
--------------------
All dependencies can be found in requirements.txt and are automatically installed when opening the Dockerfile.


SETUP
--------------------
Run main.py to preprocess and enrich event log, calculate anomaly score, and run feedback loop.


GITHUB
--------------------


Register name/email:            git config --global user.name/email "name"
Initiate git:                   git init
Connect to github repo:         git remote add origin %repo%
Adding changes to commit:       git add .
Commit changed:                 git commit -m "name or push"
Pushing changes to github:      git push -u origin main OR git push -f origin main to force changes
Clone repo:                     git clone %repo%
Stash updates if unrelated:     git stash
Fetch updates from repo:        git pull origin main --allow-unrelated-histories
Make sure you're on main:       git branch (has to be main), if not: git checkout -b main
Accept incoming chabges:        cmnd + shirt + p -> Merge Conflict: Accept All incoming

Repo for AAD:                   https://github.com/rubenpost/Model_agnostic_AAD.git
