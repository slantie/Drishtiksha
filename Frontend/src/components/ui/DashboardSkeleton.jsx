// Frontend/src/pages/DashboardSkeleton.jsx (New File)
// import React from 'react';
// import { SkeletonCard } from '../components/ui/SkeletonCard';

const DashboardSkeleton = () => (
  <div className="space-y-6 w-full max-w-full mx-auto animate-pulse">
    {/* PageHeader Skeleton */}
    <div className="h-24 bg-light-background dark:bg-dark-muted-background rounded-xl"></div>
    {/* StatCards Skeleton */}
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <div className="h-32 bg-light-background dark:bg-dark-muted-background rounded-xl"></div>
      <div className="h-32 bg-light-background dark:bg-dark-muted-background rounded-xl"></div>
      <div className="h-32 bg-light-background dark:bg-dark-muted-background rounded-xl"></div>
      <div className="h-32 bg-light-background dark:bg-dark-muted-background rounded-xl"></div>
    </div>
    {/* Search/Filter Skeleton */}
    <div className="h-20 bg-light-background dark:bg-dark-muted-background rounded-xl"></div>
    {/* DataTable Skeleton */}
    <div className="h-[500px] bg-light-background dark:bg-dark-muted-background rounded-xl"></div>
  </div>
);

export default DashboardSkeleton;