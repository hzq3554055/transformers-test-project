# Pre-commit Hook Management

This document explains how to manage the pre-commit hook for the transformers testing framework.

## 🚫 Disable Linting on Git Commit

The pre-commit hook has been **DISABLED** to allow commits without linting checks.

### Current Status
- ✅ **Pre-commit hook: DISABLED**
- ✅ **You can now commit without linting checks**

## 🔧 Available Commands

### Toggle Pre-commit Hook
```bash
# Toggle between enabled/disabled
./scripts/toggle_pre_commit.sh
```

### Enable Pre-commit Hook
```bash
# Re-enable linting checks
./scripts/enable_pre_commit.sh
```

### Check Status
```bash
# Check current status
ls -la .git/hooks/pre-commit*
```

## 📋 What the Pre-commit Hook Does

When **enabled**, the pre-commit hook runs:
1. **Linting** (flake8) - Code style checks
2. **Type checking** (mypy) - Type annotation checks  
3. **Tests** (pytest) - Unit test suite

## 🎯 When to Use Each Mode

### Disabled Mode (Current)
- ✅ **Development** - Quick commits during development
- ✅ **Experimental** - Testing new features
- ✅ **Prototyping** - Rapid iteration
- ⚠️ **Warning** - Code quality not enforced

### Enabled Mode
- ✅ **Production** - Before merging to main
- ✅ **Code review** - Before submitting PRs
- ✅ **Release** - Before tagging versions
- ✅ **Quality** - Enforces code standards

## 🚀 Quick Commands

```bash
# Disable linting (current state)
mv .git/hooks/pre-commit .git/hooks/pre-commit.disabled

# Enable linting
mv .git/hooks/pre-commit.disabled .git/hooks/pre-commit

# Check status
ls -la .git/hooks/pre-commit*
```

## 📝 Notes

- The hook is currently **DISABLED** for development convenience
- Re-enable before pushing to production
- Use `./scripts/toggle_pre_commit.sh` for easy switching
- All scripts are in the `scripts/` directory
