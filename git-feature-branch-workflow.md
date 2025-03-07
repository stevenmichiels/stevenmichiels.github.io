# Git Feature Branch Workflow

This guide outlines the proper way to create a feature in a branch and then integrate it into the main branch and push to origin.

## 1. Start with an updated main branch

```bash
# Make sure you're on the main branch
git checkout main

# Pull the latest changes
git pull origin main
```

## 2. Create a new feature branch

```bash
# Create and switch to a new feature branch
git checkout -b feature/etf-forecast
```

## 3. Work on your feature

Make your changes to the code. For your ETF Forecast website, this would include:
- Creating/modifying index.html
- Adding the Plotly.js visualization
- Any other related files

## 4. Commit your changes regularly

```bash
# Add your changes
git add index.html

# Commit with a descriptive message
git commit -m "Add ETF forecast visualization with Plotly.js"
```

## 5. Push your feature branch to origin (optional but recommended)

```bash
# Push your feature branch to the remote repository
git push -u origin feature/etf-forecast
```

## 6. Keep your feature branch updated with main (if needed)

If the main branch gets updated while you're working on your feature:

```bash
# Switch to main and pull latest changes
git checkout main
git pull origin main

# Switch back to your feature branch
git checkout feature/etf-forecast

# Merge changes from main
git merge main

# Resolve any conflicts if they occur
```

## 7. Integrate your feature into main

Once your feature is complete and tested:

```bash
# Switch to main
git checkout main

# Merge your feature branch into main
git merge feature/etf-forecast

# Resolve any conflicts if they occur
```

## 8. Push changes to origin

```bash
# Push the updated main branch to origin
git push origin main
```

## 9. Clean up (optional)

After your feature is successfully merged:

```bash
# Delete the local feature branch
git branch -d feature/etf-forecast

# Delete the remote feature branch (if you pushed it)
git push origin --delete feature/etf-forecast
```

## Alternative: Using Pull Requests (Recommended for team projects)

Instead of directly merging your feature branch into main (step 7), you can:

1. Push your feature branch to origin
2. Create a pull request on GitHub/GitLab/etc.
3. Have team members review your code
4. Merge the pull request through the web interface
5. Pull the updated main branch to your local repository

This approach provides better visibility and code review opportunities for team collaboration. 