import { Card, CardContent, CardHeader } from './Card';

const ProfileSkeleton = () => {
  const SkeletonBar = ({ className }) => (
    <div className={`bg-light-hover dark:bg-dark-hover rounded-md ${className}`} />
  );

  return (
    <div className="space-y-6 w-full max-w-full mx-auto animate-pulse">
      {/* PageHeader Skeleton */}
      <Card className="p-4 flex items-center justify-between">
        <div className="space-y-2">
          <SkeletonBar className="h-7 w-48" />
          <SkeletonBar className="h-4 w-64" />
        </div>
        <div className="flex gap-2">
          <SkeletonBar className="h-9 w-36 rounded-full" />
          <SkeletonBar className="h-9 w-32 rounded-full" />
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column Skeletons */}
        <div className="lg:col-span-1 space-y-6">
          {/* AvatarCard Skeleton */}
          <Card>
            <CardContent className="p-8 space-y-4 flex flex-col items-center">
              <div className="w-32 h-32 rounded-full bg-light-hover dark:bg-dark-hover" />
              <div className="space-y-2 w-full">
                <SkeletonBar className="h-6 w-1/2 mx-auto" />
                <SkeletonBar className="h-4 w-2/3 mx-auto" />
              </div>
              <SkeletonBar className="h-6 w-1/4 mx-auto rounded-full" />
            </CardContent>
          </Card>

          {/* Account Info Card Skeleton */}
          <Card>
            <CardHeader>
              <SkeletonBar className="h-6 w-1/3" />
            </CardHeader>
            <CardContent>
              <SkeletonBar className="h-4 w-1/2" />
              <SkeletonBar className="h-5 w-3/4 mt-1" />
            </CardContent>
          </Card>
        </div>

        {/* Right Column Skeleton */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <SkeletonBar className="h-6 w-1/3" />
            </CardHeader>
            <CardContent className="grid sm:grid-cols-2 gap-6">
              <div className="space-y-1">
                <SkeletonBar className="h-4 w-1/2" />
                <SkeletonBar className="h-5 w-3/4" />
              </div>
              <div className="space-y-1">
                <SkeletonBar className="h-4 w-1/2" />
                <SkeletonBar className="h-5 w-3/4" />
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ProfileSkeleton;