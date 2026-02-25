import Sidebar from './components/layout/Sidebar';
import PageContainer from './components/layout/PageContainer';

export default function App() {
  return (
    <div className="flex min-h-screen bg-[#030712]">
      <Sidebar />
      <PageContainer />
    </div>
  );
}
