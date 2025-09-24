#!/bin/bash
# Bull Machine Git Workflow Automation
# Usage: ./git-workflow.sh <action> [branch-name]

set -e

case "$1" in
    "new-branch")
        if [ -z "$2" ]; then
            echo "Usage: ./git-workflow.sh new-branch <branch-name>"
            exit 1
        fi
        echo "🔄 Creating new branch: $2"
        git checkout main
        git pull origin main
        git checkout -b "$2"
        git push -u origin "$2"
        echo "✅ Branch $2 created and pushed to GitHub"
        ;;

    "merge-and-new")
        if [ -z "$2" ]; then
            echo "Usage: ./git-workflow.sh merge-and-new <new-branch-name>"
            exit 1
        fi
        CURRENT=$(git branch --show-current)
        echo "🔄 Merging $CURRENT to main and creating $2"

        # Push current branch
        git push origin "$CURRENT"

        # Switch to main and merge
        git checkout main
        git pull origin main
        git merge "$CURRENT"
        git push origin main

        # Create new branch
        git checkout -b "$2"
        git push -u origin "$2"

        echo "✅ Merged $CURRENT → main, created new branch: $2"
        ;;

    "pr-ready")
        CURRENT=$(git branch --show-current)
        echo "🔄 Preparing PR for $CURRENT"
        git push origin "$CURRENT"
        echo "✅ Branch $CURRENT ready for PR"
        echo "📝 Create PR at: https://github.com/rayger14/Bull-machine-/compare/$CURRENT?expand=1"
        ;;

    "sync")
        echo "🔄 Syncing with remote"
        git fetch --all
        git status
        ;;

    *)
        echo "Bull Machine Git Workflow"
        echo "Usage:"
        echo "  ./git-workflow.sh new-branch <name>     # Create new branch from main"
        echo "  ./git-workflow.sh merge-and-new <name>  # Merge current → main, create new branch"
        echo "  ./git-workflow.sh pr-ready              # Push current branch for PR"
        echo "  ./git-workflow.sh sync                  # Sync with remote"
        ;;
esac