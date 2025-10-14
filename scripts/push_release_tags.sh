#!/bin/bash
# Push Release Tags Script
# This script helps you push all created release tags to the remote repository

set -e

echo "========================================="
echo "Push Release Tags to Remote"
echo "========================================="
echo ""

cd "$(dirname "$0")/.."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check if tags exist
TAGS=$(git tag -l "v*" | sort -V)
if [ -z "$TAGS" ]; then
    echo "❌ Error: No version tags found"
    echo "Please run ./scripts/create_release_tags.sh first"
    exit 1
fi

echo "Found the following version tags:"
echo "$TAGS"
echo ""

read -p "Do you want to push these tags to remote? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Pushing tags to remote repository..."
echo ""

# Try to push all tags
if git push origin --tags; then
    echo ""
    echo "✅ Successfully pushed all tags to remote!"
    echo ""
    echo "Next steps:"
    echo "1. Visit https://github.com/williamxie1989/stock-evaluation/tags to view tags"
    echo "2. Create GitHub Releases at https://github.com/williamxie1989/stock-evaluation/releases/new"
    echo ""
else
    echo ""
    echo "⚠️  Failed to push tags using 'git push origin --tags'"
    echo ""
    echo "You can try pushing tags individually:"
    echo ""
    for tag in $TAGS; do
        echo "  git push origin $tag"
    done
    echo ""
    exit 1
fi
