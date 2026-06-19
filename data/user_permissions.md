# Role-Based Access Control (RBAC) and Permissions

The platform uses a role-based model to manage user permissions.

## Platform Roles
1. **Owner**: Full administrative control, including billing setup, account deletion, and user role modification.
2. **Administrator**: Can add/remove users, configure APIs, adjust integration settings, but cannot delete the workspace owner.
3. **Editor**: Can read and write database integrations, webhooks, and setup pipelines. Cannot modify users.
4. **Viewer**: Read-only access to dashboard data and analytics.

## Modifying Permissions
To upgrade a user's permission level:
1. Navigate to **Admin Panel > Team Members**.
2. Locate the user and click **Edit Permissions**.
3. Select the new Role from the dropdown list.
4. Click **Save Changes**. The permissions update takes effect on the user's next API request or page load.
