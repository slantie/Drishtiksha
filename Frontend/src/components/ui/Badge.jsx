// src/components/ui/Badge.jsx

import React from "react";
import { cn } from "../../utils/cn";
import {
  badgeVariants,
  statusConfigs,
  mediaTypeConfigs,
  deviceConfigs,
} from "../../constants/badgeConstants";

export const Badge = React.forwardRef(
  (
    {
      children,
      className,
      variant,
      size,
      rounded,
      animate,
      icon: Icon,
      iconSize = "w-3.5 h-3.5",
      status,
      mediaType,
      device,
      version,
      color,
      capitalize = false,
      uppercase = false,
      mono = false,
      ...props
    },
    ref
  ) => {
    // Determine configuration based on type
    let config = {};
    let content = children;

    // Status badge
    if (status && statusConfigs[status]) {
      config = statusConfigs[status];
      content = content || config.label;
      capitalize = true;
    }

    // Media type badge
    if (mediaType && mediaTypeConfigs[mediaType]) {
      config = mediaTypeConfigs[mediaType];
      content = content || config.label;
    }

    // Device badge
    if (device && deviceConfigs[device]) {
      config = deviceConfigs[device];
      content = content || config.label;
      uppercase = true;
      mono = true;
    }

    // Version badge
    if (version) {
      config = { variant: "primary" };
      content = content || (version === "N/A" ? "N/A" : `Version ${version}`);
      mono = true;
    }

    // Color override
    if (color) {
      const colorMap = {
        green: "success",
        red: "danger",
        yellow: "warning",
        blue: "info",
        purple: "purple",
        indigo: "indigo",
        pink: "pink",
        gray: "default",
        primary: "primary",
      };
      config.variant = colorMap[color] || color;
    }

    // Merge configurations
    const finalVariant = variant || config.variant || "default";
    const finalAnimate = animate || config.animate || "none";

    return (
      <span
        ref={ref}
        className={cn(
          badgeVariants({
            variant: finalVariant,
            size,
            rounded,
            animate: finalAnimate,
          }),
          capitalize && "capitalize",
          uppercase && "uppercase",
          mono && "",
          className
        )}
        {...props}
      >
        {Icon && <Icon className={cn(iconSize, "flex-shrink-0")} />}
        {content}
      </span>
    );
  }
);

Badge.displayName = "Badge";

// Convenience components for common use cases
export const StatusBadge = ({ status, ...props }) => (
  <Badge status={status} {...props} />
);

export const MediaTypeBadge = ({ mediaType, icon: Icon, ...props }) => (
  <Badge mediaType={mediaType} icon={Icon} {...props} />
);

export const DeviceBadge = ({ device, ...props }) => (
  <Badge device={device} size="sm" {...props} />
);

export const VersionBadge = ({ version, ...props }) => (
  <Badge version={version} size="sm" {...props} />
);
