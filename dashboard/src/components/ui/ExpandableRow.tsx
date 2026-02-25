import { ChevronDown } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

interface ExpandableRowProps {
  header: React.ReactNode;
  children: React.ReactNode;
  expanded: boolean;
  onToggle: () => void;
}

export default function ExpandableRow({ header, children, expanded, onToggle }: ExpandableRowProps) {
  return (
    <div className="border-b border-white/[0.05] last:border-b-0">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/[0.03] transition-colors"
      >
        <div className="flex-1 text-left">{header}</div>
        <ChevronDown
          className={`w-4 h-4 text-slate-500 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
        />
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-1">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
